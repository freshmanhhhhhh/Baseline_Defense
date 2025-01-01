# Baseline Defenses for Adversarial Attacks Against Aligned Language Models
Reproduce Code for "Baseline Defenses for Adversarial Attacks Against Aligned Language Models"


## Datasets
Using [jade-db-v2](https://github.com/whitzard-ai/jade-db/tree/main/jade-db-v2.0) as unsafe prompts, and we applied jailbreak attacks to jade-db-v2,  including DeepInception and ReNeLLM. Then we get jailbroken prompts jade-inception and rene_llm.
Add zhihu+rlhf_zh as normal prompts.


## baseline defenses

### Perplexity Filter

The perplexity filter in the code consists of two filters, a perplexity filter which as also been proposed in concurrent work by [Alon et al.](https://arxiv.org/abs/2308.14132) and a windowed perplexity filter, which consists of checking the perplexity of a window of $n$ tokens.

### Paraphrase Defense

The paraphrase defense is rewriting the prompt. For our experiments, we used ChatGPT. Note while this defense is effective it might come at high performance cost.

### demo
Test on Chinese dataset jade-db-v2 easy version and medium version, as well as these prompts with jailbreak attacks including [DeepInception](https://github.com/tmlr-group/DeepInception),[ReNeLLM](https://github.com/NJUNLP/ReNeLLM),[GCG](https://github.com/llm-attacks/llm-attacks).

## llm as classifier

Applied models: [QWEN2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), [Llama-Guard3-1B](https://huggingface.co/meta-llama/Llama-Guard-3-1B), [Prompt-Guard3-86M](https://huggingface.co/meta-llama/Prompt-Guard-86M)
