# 数据版本标注（DATA_VERSIONS.md）

> 更新时间：2026-08-17
> 本文件记录 `data/` 目录下各数据集的版本、来源、格式与用途，便于复现与追溯。

## 1. 原始诗集（风格池 / Pretrain 数据）

| 文件 | 版本 | 条数 | 格式 | 来源 | 用途 |
|---|---|---|---|---|---|
| `gucheng_train.jsonl` | v1 | 183 | ShareGPT（system/human/gpt） | 顾城诗集 | 顾城风格池、SFT/KTO desirable 源 |
| `haizi_train.jsonl` | v1 | 约 120 | ShareGPT | 海子诗集 | 海子风格池 |
| `haizi_cn_train.jsonl` | v1 | 约 120 | ShareGPT | 海子（CN 版） | 风格池 |
| `libai_train.jsonl` | v1 | 838 | ShareGPT | 李白诗集 | 李白风格池、SFT 训练 |
| `libai_train_300.jsonl` | v1.1 | 300 | ShareGPT | 从 libai_train 取前 300 | 快速 SFT 子集 |
| `gucheng_train_annotated.jsonl` | v1 | 少量 | ShareGPT+标注 | 人工标注 | 标注补充 |
| `haizi_train_annotated.jsonl` | v1 | 约 60 | ShareGPT+标注 | 人工标注 | 标注补充 |
| `libai_train_annotated.jsonl` | v1 | 少量 | ShareGPT+标注 | 人工标注 | 标注补充 |
| `style100_titles.jsonl` | v1 | 100 | JSONL | 标题池 | 生成输入标题 |

## 2. 标注一致数据集（人工多标注）

| 文件 | 版本 | 条数 | 说明 |
|---|---|---|---|
| `annotations_consistent.jsonl` | v1 | 396（诗 182 / 非诗 214） | 多位标注者一致：is_poetry 标签 + text |
| `annotations_dpo_poem_nonpoem.jsonl` | v1 | 182 | 诗样本偏好对（chosen=诗, rejected=非诗） |
| `annotations_dpo_poem_vs_nonpoem.jsonl` | v1 | 182 | 同上（判别式 prompt 版本） |

## 3. SFT / DPO / KTO 训练数据

| 文件 | 版本 | 条数 | 格式 | 用途 |
|---|---|---|---|---|
| `annotations_sft_style.jsonl` | v1 | 180（顾113/海64/李3） | conversations + chosen/rejected | SFT 风格数据 |
| `annotations_dpo_gen_format.jsonl` | v1 | 180 | conversations + chosen(真诗) + rejected(非诗) + meta.style | DPO/KTO 偏好数据 |
| `annotations_dpo_gen_libai.jsonl` | v1.1 | 838 | 同上 | 李白 KTO 补充数据（新增） |
| `poetry_dpo_qwen38.jsonl` | v1 | 633KB | DPO | Qwen3.8 DPO 数据 |

## 4. Hard 难样本生成（论文核心）

| 文件 | 版本 | 条数 | 说明 |
|---|---|---|---|
| `hard_dataset_5000.jsonl` | v1 | 5000 | 论文基线 hard 数据集 |
| `hard_dataset_v2_4343.jsonl` | v2 | 4343 | v2 hard 数据集 |
| `hard_dataset_qwen38_500.jsonl` | v1 | 500 | Qwen3.8 生成 500 条 |
| `hard_input_1000.jsonl` | v1 | 1000 | 1000 条生成输入（Haizi/GuCheng/Haizi-CN/LiBai 各 250） |
| `hard_gen_GuCheng.jsonl` / `-Haizi.jsonl` / `-Haizi-CN.jsonl` / `-LiBai.jsonl` | v1 | 各约 250 | 各风格生成结果 |
| `hard_gen_v2_*.jsonl` | v2 | 各约 250 | v2 生成结果 |
| `hard_gen_7b_sft_300.jsonl` | v1 | 300 | 7B SFT 生成 |
| `hard_stats_*.json` | v1/v2 | - | 生成统计 |
| `v2_pool_*.jsonl` | v2 | 各约 60 | v2 风格池 |

## 5. 其他

| 文件 | 说明 |
|---|---|
| `baseline_random.json` | 随机基线统计 |
| `recited_removed.jsonl` | 去除背诵过的样本 |
| `to_annotate_near.jsonl` / `to_annotate_report.md` | 待标注候选与报告 |

## 版本变更记录

- **2026-08-16**：新增 `annotations_dpo_gen_libai.jsonl`（李白 KTO 838 条）、`libai_train_300.jsonl`（快速 SFT 300 条）、`hard_input_1000.jsonl`（1000 输入）
- **2026-08-15**：新增 `annotations_consistent.jsonl`、`annotations_sft_style.jsonl`、`annotations_dpo_*` 标注一致数据
- 更早：`hard_dataset_v2_*` 系列（v2 难样本）

## 关联仓库

- GitHub 主仓库：https://github.com/shikunpneg/ChineseHardJudgePoem
- MiniMind 顾城生成器仓库：https://github.com/shikunpneg/GuCheng-POEM
- HuggingFace 数据集：https://huggingface.co/shikunpunk/gucheng-poetry-dataset
- 16 区云盘：`/matpilot/datasets/gucheng_poem_data_20260817/`
