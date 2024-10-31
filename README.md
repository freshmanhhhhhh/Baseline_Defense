# Baseline Defenses for Adversarial Attacks Against Aligned Language Models
Reproduce Code for "Baseline Defenses for Adversarial Attacks Against Aligned Language Models"

## Perplexity Filter

The perplexity filter in the code consists of two filters, a perplexity filter which as also been proposed in concurrent work by [Alon et al.](https://arxiv.org/abs/2308.14132) and a windowed perplexity filter, which consists of checking the perplexity of a window of $n$ tokens.

## Paraphrase Defense

The paraphrase defense is rewriting the prompt. For our experiments, we used ChatGPT. Note while this defense is effective it might come at high performance cost.

## demo
Test on Chinese dataset jade-db-v2 easy version and medium version, as well as these prompts with jailbreak attacks including [Cipher](https://github.com/RobustNLP/CipherChat),[DeepInception](https://github.com/tmlr-group/DeepInception),[ReNeLLM](https://github.com/NJUNLP/ReNeLLM),[GCG](https://github.com/llm-attacks/llm-attacks).