# ChineseHardJudgePoem 中文诗歌 Hard 难样本数据集

一个用于**中文诗歌真伪判别**与**风格保真度评估**的大规模 Hard 难样本数据集：由四个 QLoRA 微调的中文诗歌生成模型（李白 / 海子 / 海子-中文约束 / 顾城）模仿真实诗歌数据集创作，共 **5000 条**带相似度标注的样本。

[研究论文 (中文)](paper/research_paper.md) · [相似度统计报告](hard_data/hard_dataset_similarity_report.md)

---

## 1. 项目简介

- **任务**：构建"最像真实诗歌"的模型生成样本，用于测试/训练诗歌真伪判别器（poetry-judge）。
- **规模**：5000 条样本，四个风格模型各 1250 条。
- **标注**：每条含字符 2-gram Jaccard / 余弦相似度（vs 同标题真实诗）与风格池平均相似度。
- **基座**：Qwen2.5-3B-Instruct，QLoRA 4-bit 微调（RTX 4060 Laptop 8GB 可训练）。

## 2. Hard 数据集创建过程

### 2.1 提示词池

从四个训练集（`libai_train` / `haizi_train` / `haizi_cn_train` / `gucheng_train`）抽取唯一标题，合并为统一标题池（309 个），每个模型按训练时一致的提示模板构造生成请求：

- 李白：`你是诗人李白，请以《{title}》为题，写一首你觉得好的诗歌。`
- 海子 / 海子-CN / 顾城：`请以《{title}》为题，创作一首现代诗。`

### 2.2 模型生成

- 四个模型轮流创作，每模型 1250 首，合计 5000 首；
- 解码配置：`temperature=1.0, top_p=0.9, repetition_penalty=1.05, max_new_tokens=200`；
- 固定随机种子 `20260812`，采样全流程可复现；
- 4-bit 量化加载，`batch_size=6`，RTX 4060 Laptop 8GB 批量生成。

### 2.3 相似度计算

| 指标 | 定义 |
|------|------|
| `sim_jaccard` | 与同标题真实诗的字符 2-gram 集合 Jaccard 相似度 |
| `sim_cosine` | 与同标题真实诗的字符 2-gram 词频余弦相似度 |
| `sim_pool` | 与该模型风格真实诗池随机 20 首的平均 Jaccard 相似度 |

### 2.4 数据格式

```json
{
  "model": "GuCheng",
  "title": "悟",
  "genre": "现代诗",
  "prompt": "请以《悟》为题，创作一首现代诗。",
  "generated": "……生成诗歌正文……",
  "real_text": "……同标题真实诗歌……",
  "sim_jaccard": 0.0123,
  "sim_cosine": 0.0456,
  "sim_pool": 0.0102
}
```

## 3. 诗歌生成模型技术原理

### 3.1 基座与量化

统一采用 **Qwen2.5-3B-Instruct** 为基座，通过 **QLoRA** 在 4-bit（NF4）量化模型上训练低秩适配器，适配 8GB 显存单卡：

```
quantization:  bitsandbytes NF4 4-bit, bf16 compute, double quant
LoRA:          rank=32, alpha=64, dropout=0.05, target=all
训练:          SFT, lr=2e-4, cosine, warmup=0.1, grad_accum=8
```

### 3.2 数据与训练轮数

| 模型 | 数据来源 | 样本数 | 轮数 |
|------|----------|-------|------|
| LiBai | 《李太白全集》注本诗歌（截断至 3686 行排除附录） | 838 | 10 |
| Haizi | 海子诗集 | 134 | 10 |
| Haizi-CN | 海子诗集 + 中英文混入负面样本 + 中文约束样本 | 155 | 3 |
| GuCheng | 顾城诗集 | 183 | 8 |

### 3.3 中文约束（Haizi-CN）

针对高温采样（T≥1.8）下模型混入英文的问题，构建三类负面样本：

1. `negative_english_mix.jsonl`：英文片段自动检测与定位；
2. `dpo_preference.jsonl`：纯中文（chosen）/ 混英文（rejected）偏好对；
3. `sft_cn_constraint.jsonl`：提示内嵌硬约束 *"注意：必须全部使用中文，禁止混入任何英文字母、单词或句子。"*

### 3.4 模型发布

| 风格 | Hugging Face |
|------|--------------|
| 李白 | [shikunpunk/Qwen2.5-3B-LiBai](https://huggingface.co/shikunpunk/Qwen2.5-3B-LiBai) |
| 海子 | [shikunpunk/Qwen2.5-3B-Haizi](https://huggingface.co/shikunpunk/Qwen2.5-3B-Haizi) |
| 海子-中文约束 | [shikunpunk/Qwen2.5-3B-Haizi-CN](https://huggingface.co/shikunpunk/Qwen2.5-3B-Haizi-CN) |
| 顾城 | [shikunpunk/Qwen2.5-3B-GuCheng](https://huggingface.co/shikunpunk/Qwen2.5-3B-GuCheng) |

## 4. 目录结构

```
.
├── README.md
├── data/
│   └── hard_dataset_5000.jsonl      # 5000 条 Hard 难样本数据集
├── hard_data/
│   ├── hard_gen_GuCheng.jsonl       # 四模型分组生成结果
│   ├── hard_gen_Haizi.jsonl
│   ├── hard_gen_Haizi-CN.jsonl
│   ├── hard_gen_LiBai.jsonl
│   ├── hard_stats_*.json            # 各组相似度统计
│   └── hard_dataset_similarity_report.md
├── paper/
│   ├── research_paper.md            # 中文研究论文
│   └── figures/                     # 论文图表
└── scripts/
    ├── build_hard_inputs.py         # 构建 5000 条生成输入
    ├── gen_hard.py                  # 四模型轮流生成 + 相似度标注
    └── make_hard_dataset.py         # 合并数据集 + 统计报告
```

## 5. 复现指南

```bash
# 1. 下载模型（Hugging Face，4 个仓库）
# 2. 构建生成输入
python scripts/build_hard_inputs.py
# 3. 四模型轮流生成 + 相似度标注
python scripts/gen_hard.py --model models/Qwen2.5-3B-GuCheng  --group GuCheng
python scripts/gen_hard.py --model models/Qwen2.5-3B-Haizi    --group Haizi
python scripts/gen_hard.py --model models/Qwen2.5-3B-Haizi-CN --group "Haizi-CN"
python scripts/gen_hard.py --model models/Qwen2.5-3B-LiBai-Final --group LiBai
# 4. 合并数据集 + 报告
python scripts/make_hard_dataset.py
```

环境要求：Python 3.10+，`transformers`、`peft`、`bitsandbytes`、`torch`（CUDA）。

## 6. 引用

```bibtex
@misc{chinesehardjudgepoem2026,
  title={ChineseHardJudgePoem: A Hard Dataset for Chinese Poetry Authenticity Discrimination},
  author={shikunpunk},
  year={2026},
  howpublished={\url{https://github.com/shikunpneg/ChineseHardJudgePoem}}
}
```
