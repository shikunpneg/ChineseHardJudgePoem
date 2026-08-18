# 余华作品数据与风格模型（MiniMind 104M）

来源：《余华作品全集（全13册）》epub 解析。正文按顺序拆分为段落，标注册/章节/段落顺序。

## 数据
| 文件 | 说明 |
|---|---|
| `yuhua_raw_full.jsonl` | 13 册全文分段（20952 段），字段 `{book, book_index, chapter, chapter_index, segment_index, text}` |
| `pretrain_yuhua.jsonl` | 预训练数据（18793 条，`{"text": ...}`） |
| `sft_yuhua_cot.jsonl` | 余华风格 CoT SFT 数据（522 条）：【构思】→【基调】→【正文】 |
| `yuhua_samples_cot_v1.jsonl` | 100 条生成样本（`full_sft_yuhua_cot`） |

## 脚本
- `parse_yuhua_epub.py` — epub 解析（spine 顺序 + ncx 树定位 13 册边界）
- `build_yuhua_dataset.py` — 构建 pretrain / CoT SFT 数据
- `gen_yuhua_batch.py` — 余华风格批量生成

## 训练
1. pretrain（warm-start 自顾城权重）：`train_pretrain.py --save_weight pretrain_yuhua --data_path ../dataset/pretrain_yuhua.jsonl --from_weight pretrain_gucheng --num_workers 0`
2. SFT（CoT）：`train_full_sft.py --save_weight full_sft_yuhua_cot --data_path ../dataset/sft_yuhua_cot.jsonl --from_weight pretrain_yuhua --num_workers 0`

注意：Windows 下 DataLoader 必须 `--num_workers 0`，否则训练卡死。

## 局限（诚实记录）
104M 模型 + 522 条 CoT 数据能稳定输出【构思】【基调】【正文】结构，但正文长文连贯性有限，常出现荒诞拼贴。更大基座（7B LoRA）+ 更多 CoT 数据是改进方向。
