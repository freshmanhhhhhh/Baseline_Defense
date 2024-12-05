# nohup python judgebyLG3.py --ratio balance --harm_dset jade > ./logs/LlamaGuard3-1B/combine_jade_benchmark_easy_1k1k.log 2>&1 & 

import torch
import os
import json
import pandas as pd
import random
import time
from datetime import datetime
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 定义命令行参数
parser = argparse.ArgumentParser(description="控制样本比例和 system_prompt 类型")
parser.add_argument('--ratio', type=str, choices=['balance', 'real'], required=True,
                   help="正负样本比例，支持平衡比例 '1:1' 或接近真实场景负样本更少的比例 '4:1'")
# parser.add_argument('--sysprompt_type', type=str, choices=['no_type', 'type'], required=True,
#                     help="选择 system_prompt 类型，支持'no_type'仅分类是否有害，或 'type'分类并描述有害类型及是否越狱")
parser.add_argument('--harm_dset', type=str, choices=['jade','inception','renellm'], required=True,
                    help='选择恶意样本，jade为基准恶意提问，inception为jade嵌套到梦境中，renellm为jade应用renellm越狱攻击')
# parser.add_argument("--classifier", type=str, choices=['qwen2.5-1.5b', 'prompt-guard-86M','llama-guard3-1b'], required=True,
#                     help="选择用于分类的大模型的类型")
args = parser.parse_args()

# load model
model_name = "Llama-Guard-3-1B"
local_model_path = "/home/yangshuhan/data/BackupRestore/data/llm_sec/models/Llama-Guard-3-1B"
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True
)
tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)


# 读取待检测的提示问题
def read_q_csv(input_file) -> list:
    # 读取CSV文件  
    df = pd.read_csv(input_file)    
    
    # 提取“问题”列的内容    
    user_sequences = df['问题'].tolist()
    return user_sequences

def read_q_tsv(input_file) -> list:
    df = pd.read_csv(input_file, sep = '\t')    
    
    # 提取“问题”列的内容    
    user_sequences = df['prompt'].tolist()
    return user_sequences

def read_q_json(input_file) -> list:
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 提取所有 'rewritten_prompt' 的值
    user_sequences = [item['nested_prompt'] for item in data]
    return user_sequences

# combine neg_prompt and pos_prompt
# read neg_prompts
# 先只考虑easy
if args.harm_dset == 'renellm':
    input_file = "./datasets/rene_llm/renellm_judge_by_qwen2_attack_on_llama3_chinese_easy_2.0.json" 
    neg_prompts = read_q_json(input_file)
elif args.harm_dset == 'inception':
    input_file = './datasets/jade_inception/jade_inception_easy.csv'
    neg_prompts = read_q_csv(input_file)
else:
    input_file = './datasets/jade-db-v2/jade_benchmark_easy_zh.csv'
    neg_prompts = read_q_csv(input_file)

# read pos_prompts
input_file = './datasets/zhihu_rlhf_zh/zhihu_rlhf_1k_noRepeat.tsv'
pos_prompts = read_q_tsv(input_file)

# combine neg and pos prompts according to the ratio
if args.ratio == 'real':
    neg_cnt = min(len(neg_prompts), min(len(pos_prompts),1000)//4)
    pos_cnt = neg_cnt*4
else: # balance as default
    neg_cnt = pos_cnt = min(len(neg_prompts),len(pos_prompts))
print("neg_cnt:{}, pos_cnt:{}".format(neg_cnt, pos_cnt))
# random sample prompts in both neg_prompts and pos_prompts
selected_pos = random.sample(pos_prompts, pos_cnt)
selected_neg = random.sample(neg_prompts, neg_cnt)
prompts = selected_neg + selected_pos
# label prompts
neg_labels = [1] * len(selected_neg)
pos_labels = [0] * len(selected_pos)
labels = neg_labels + pos_labels
# shuffle prompts and labels
combined = list(zip(prompts, labels))
random.shuffle(combined)
# 拆分回 prompts 和 labels
shuffled_prompts, shuffled_labels = zip(*combined)

# 转换为列表形式
prompts = list(shuffled_prompts)
y_labels = list(shuffled_labels)

y_pred = []
results = []
toxicity_cnt = 0
time_costs = []
sum_len = 0

# use llm to classify prompts
for p in prompts:
    # 计算判断一条提示的时间开销
    start_time = time.time()
    
    sum_len = sum_len + len(p)
    
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": p
                },
            ],
        }
    ]
    
    input_ids = tokenizer.apply_chat_template(
        conversation, return_tensors="pt"
    ).to(model.device)

    prompt_len = input_ids.shape[1]
    output = model.generate(
        input_ids,
        max_new_tokens=20,
        pad_token_id=0,
    )
    generated_tokens = output[:, prompt_len:]
    response = tokenizer.decode(generated_tokens[0]).strip()
    
    print("User prompt:", p)
    print("Response from", model_name.split("/")[-1],":\n", response)
    
    # 解析响应
    response = response.strip().replace('<|eot_id|>', '').split('\n')
    while len(response) < 2:
        response.append("缺少响应行")
    if response[0]=='safe':
        toxicity_classification = 0
    elif response[0]=='unsafe':
        toxicity_classification = 1
    else:
        print("unexpected response!")
        exit(1)
    toxicity_type = response[1]

    # 构建结果记录
    result = {
        "prompt": p,
        "toxicity_classification": toxicity_classification,
        "toxicity_type": toxicity_type
    }
    results.append(result)
    
    y_pred.append(toxicity_classification)
    
    end_time = time.time()
    
    duration = end_time - start_time
    print("process time: {}".format(duration))
    
    time_costs.append(duration)

# 保存结果到模型对应目录下的JSON文件中
target_dir = "./results/llm_judge/" + model_name.split("/")[-1]
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
    print("Create [", target_dir, "] successfully!")

# 获取当前日期，并构建输出文件名，包含输入文件名
current_time = datetime.now().strftime("%Y%m%d_%H%M")
input_filename = input_file.split("/")[-1].split(".")[0]
output_file = target_dir + f"/{input_filename}_{current_time}.json"

# 将结果保存到 JSON 文件中
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print("Save successfully!")

# sklearn.metrics计算分类性能指标
acc = accuracy_score(y_labels, y_pred)
precision = precision_score(y_labels, y_pred)
recall = recall_score(y_labels, y_pred)
f1 = f1_score(y_labels, y_pred)

print("[*]PERFORMANCE[*]\n\
        acc: {}\n\
        precision: {}\n\
        recall: {}\n\
        f1: {}".format(acc, precision, recall, f1))
print("[*]AVERAGE PROCESS TIME[*]\n\
        {}s with average length {}".format(sum(time_costs)/len(time_costs), sum_len/len(prompts)))