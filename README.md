# qwen-lora-tutor

基于 Qwen2.5-0.5B-Instruct 的 LoRA 微调项目 — 第5周：微调入门

## 🎯 项目简介

使用 LoRA（Low-Rank Adaptation）在 ModelScope 云端环境对 Qwen2.5-0.5B-Instruct 进行指令微调，训练一个 **AI 学习助教**，能够用通俗易懂的语言回答 AI/LLM 入门问题。

## 🧠 核心概念

### 什么是 LoRA？

LoRA 是一种**高效微调**方法，不修改原模型参数，而是在旁边挂两个小矩阵（低秩矩阵），只训练这些小矩阵。效果接近全量微调，但训练参数减少 99% 以上。

| 对比 | 全量微调 | LoRA |
|------|---------|------|
| 训练参数量 | 全部（5亿） | ~10万 |
| 显存需求 | 高 | 低 |
| 训练速度 | 慢 | 快 |
| 效果 | 最好 | 接近 |

## 📂 项目结构

```
├── adapter/                  # LoRA 适配器权重
│   ├── adapter_config.json   # LoRA 配置（r=8, alpha=16）
│   └── adapter_model.safetensors  # 训练好的 LoRA 权重 (~2MB)
├── data/
│   └── training_data.json    # 训练数据（问答对）
├── requirements.txt          # 依赖
└── README.md
```

## ⚙️ LoRA 配置

| 参数 | 值 | 说明 |
|------|-----|------|
| r | 8 | 低秩矩阵秩（秩越大表示能力越强，训练越慢） |
| lora_alpha | 16 | 缩放系数 |
| lora_dropout | 0.05 | Dropout 防止过拟合 |
| target_modules | q_proj, v_proj | 在 Attention 的 Q 和 V 上挂 LoRA |
| task_type | CAUSAL_LM | 因果语言模型任务 |
| 基座模型 | Qwen2.5-0.5B-Instruct | 5亿参数指令模型 |

## 📊 训练数据

15 条中文问答对，覆盖 AI 入门常见问题：

- AI 基础概念（机器学习、深度学习、Transformer）
- 编程入门（Python）
- 学习方法（如何学习 AI）
- BERT / GPT / LLM 概念

## 🚀 使用方法

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 加载基座模型
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# 加载 LoRA 适配器
model = PeftModel.from_pretrained(model, "./adapter")

# 推理
inputs = tokenizer("什么是 Transformer？", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))
```

## 🛠 技术栈

- **基座模型**: Qwen2.5-0.5B-Instruct（通义千问）
- **微调方法**: LoRA（peft 库）
- **运行环境**: ModelScope 云端 GPU（国内可用，无需 HuggingFace）
- **依赖**: transformers, peft, datasets, accelerate, modelscope

## 📝 学习路径

本项目是 **60 天 AI 基础搭建计划** 的第 5 周内容：

| 周次 | 主题 | 项目 | 状态 |
|------|------|------|:--:|
| 第 5 周 | 微调入门 | [qwen-lora-tutor](https://github.com/lalaland472/qwen-lora-tutor) | ✅ |
| 第 6 周 | Embedding 与向量搜索 | [semantic-search-demo](https://github.com/lalaland472/semantic-search-demo) | ✅ |
| 第 7 周 | Alignment 入门 | dpo-mini-experiment | ⬜ |
| 第 8 周 | RAG 系统 | rag-knowledge-bot | ⬜ |

## 📚 参考资料

- [LoRA 论文 (2021)](https://arxiv.org/abs/2106.09685) — Low-Rank Adaptation of Large Language Models
- [Qwen2.5 官方文档](https://github.com/QwenLM/Qwen2.5)
- [PEFT 文档](https://huggingface.co/docs/peft)
