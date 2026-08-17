# Remote Jupyter - poetry-hard

This folder contains all files from the remote Jupyter notebook machine (`px-cloud1.matpool.com`).

## Structure

```
remote-jupyter/
├── data/               # Training data
│   ├── hard_input_1000.jsonl       # 1000 input poems
│   ├── gucheng_train.jsonl         # GuCheng training data
│   ├── haizi_train.jsonl           # Haizi training data
│   ├── libai_train.jsonl           # LiBai training data
│   ├── libai_train_300.jsonl       # LiBai 300 samples
│   ├── libai_input_200.jsonl       # LiBai 200 input prompts
│   ├── annotations_dpo_gen_format.jsonl  # DPO format annotations
│   └── annotations_dpo_gen_libai.jsonl   # LiBai DPO annotations
├── scripts/            # Training & eval scripts
│   ├── kto_style.py    # KTO training
│   ├── sft_style.py    # SFT training
│   └── gen_kto_eval.py # Evaluation
├── saves/              # Model adapters & results
│   ├── sft-GuCheng-ann/    # SFT adapter (GuCheng)
│   ├── sft-Haizi-ann/      # SFT adapter (Haizi)
│   ├── sft-LiBai-LiBai/    # SFT adapter (LiBai)
│   ├── kto-GuCheng-GuCheng/  # KTO adapter (GuCheng)
│   ├── kto-Haizi-Haizi/      # KTO adapter (Haizi)
│   ├── kto-LiBai-LiBai/      # KTO adapter (LiBai)
│   ├── eval_kto_*.jsonl      # Evaluation results
│   ├── stats_kto_*.json     # Evaluation stats
│   └── gen_libai_200.jsonl  # Generated 200 LiBai poems
├── logs/               # Training logs
│   ├── kto_*.log
│   ├── sft_*.log
│   └── eval_*.log
└── root/               # Root directory scripts/logs
```

## Large Files

Files larger than 50MB (`.safetensors`, `tokenizer.json`) are not included.
Placeholder `.placeholder` files indicate where the original files were.

Model weights are available on HuggingFace:
- `shikunpunk/Qwen3.8-27B-LiBai-KTO`
