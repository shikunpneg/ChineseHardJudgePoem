---
language:
  - zh
tags:
  - poetry-generation
  - modern-poetry
  - gucheng
  - qwen
  - lora
  - peft
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
pipeline_tag: text-generation
---

# Qwen3.8-27B-GuCheng

基于 [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) 使用 **QLoRA 微调**的**顾城风格现代诗生成模型**（LoRA adapter）。模型在最新 Qwen3.8-27B 基座上学习诗人顾城的意象组合、语言节奏与精神气质，可生成短诗、中等篇幅、长诗等不同长度的现代诗。

> 本仓库保存的是 **LoRA adapter**，加载时需配合基座模型 Qwen/Qwen3.8-27B 使用。

## 模型概览

| 项目 | 说明 |
| --- | --- |
| 基础模型 | Qwen/Qwen3.8-27B（Qwen3.8，2026-08 最新版） |
| 参数量 | 27B（基座） |
| 训练方法 | QLoRA（4bit NF4 量化，rank=16, alpha=32） |
| 训练数据 | 顾城诗歌 183 首（gucheng_train.jsonl，人类标注一致样本 113 首可并入） |
| 训练硬件 | NVIDIA A100-PCIE-40GB（云服务） |
| 训练轮数 | 8 epochs，train_loss 1.69 → 0.47 |
| 关闭思考模式 | 推理时 `enable_thinking=False`，直接生成诗歌 |

## 快速使用

```python
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

model_id = "Qwen/Qwen3.8-27B"
adapter_id = "shikunpunk/Qwen3.8-27B-GuCheng"

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True,
    dtype=torch.bfloat16, device_map="auto",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                           bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True),
)
model = PeftModel.from_pretrained(model, adapter_id)
model.eval()

msgs = [
    {"role": "system", "content": "你是一位深谙顾城诗歌风格的现代诗人。顾城的诗以简洁的意象、童话般的想象和对生命本质的追问为特征，语言纯净而富有灵气，常带一丝忧郁与梦幻。"},
    {"role": "user", "content": "请以《小巷》为题，创作一首现代诗。"},
]
text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
enc = processor(text=text, return_tensors="pt").to(model.device)
out = model.generate(**enc, max_new_tokens=200, temperature=1.0, top_p=0.9,
                     repetition_penalty=1.05, do_sample=True)
print(processor.tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True))
```

## 训练方法

- 基座：Qwen/Qwen3.8-27B（最新 Qwen3.8 稠密模型，Qwen3_5ForCausalLM）
- 量化：4bit NF4（bitsandbytes），compute_dtype=bfloat16，双重量化
- LoRA：rank=16, alpha=32, dropout=0.05，target_modules=in_proj_qkv/out_proj/gate_proj/up_proj/down_proj 等全部线性层
- 训练：SFT，lr=2e-4，cosine 调度，warmup 10 步，gradient_checkpointing=True
- 有效 batch size：16（per_device=2 × grad_accum=8）
- 训练环境：A100-PCIE-40GB（云服务），关闭 thinking 模式保证直接输出诗歌正文

## 与旧版对比

| 版本 | 基座 | 参数量 | 说明 |
| --- | --- | --- | --- |
| Qwen2.5-3B-GuCheng（旧） | Qwen2.5-3B-Instruct | 3B | 本地 RTX 4060 训练 |
| **Qwen3.8-27B-GuCheng（本仓库）** | Qwen3.8-27B | 27B | A100 云服务训练，风格模仿能力更强 |

## 数据与复现

- 训练数据：`gucheng_train.jsonl`（183 首顾城诗歌，ShareGPT 格式）
- 生成脚本：`scripts/sft_style.py`（ChineseHardJudgePoem 仓库）
- 该模型用于 AI4S「中文诗歌风格保真与真伪判别」研究的困难负样本生成与风格模仿验证

## 引用

```bibtex
@misc{chinesehardjudgepoem2026,
  title={ChineseHardJudgePoem: A Hard Dataset for Chinese Poetry Authenticity Discrimination},
  author={shikunpunk},
  year={2026},
  howpublished={\url{https://github.com/shikunpneg/ChineseHardJudgePoem}}
}
```
