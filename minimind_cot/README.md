# 顾城 AR 模型思维链(CoT)优化

在最佳 AR 模型（`full_sft_gucheng`）上增加思维链后训练：**先读题→理解→转化为诗歌**。

## 模板
```
【读题】《题目》让我想到<意象>构成的画面，<情感>的气息藏在字句之间……
【意象】意象词列表
【情感基调】<情感词>
【诗】
<诗歌正文>
```

## 数据与产物
| 文件 | 说明 |
|---|---|
| `sft_gucheng_cot.jsonl` | CoT SFT 数据（607 条，CoT 前缀由真实诗文本特征规则生成，不虚构） |
| `gucheng_samples_ar_cot_full.jsonl` | 100 条生成样本（模型 `full_sft_gucheng_cot`，温度 ~1.0） |
| `build_gucheng_cot.py` | CoT 数据构建脚本 |
| `gen_gucheng_batch.py` | 批量生成脚本（含 `fix_title_drift` 解码约束：强制【读题】原样引用题目） |

## 评估
- 100/100 通过质量过滤（无英文/乱码/重复/过短）
- 模型学会 CoT 三段式结构
- 已知问题：104M 模型采样时【读题】书名号偶发漂移 → 已用解码约束修复（92/100 生成后题目一致）
- 阴性结果：指令增强 SFT v2（强化"原样引用题目"指令，6 epochs）导致 CoT 结构退化，确认解码约束是更可靠修复

## 训练命令
```
python train_full_sft.py --save_weight full_sft_gucheng_cot --epochs 10 --batch_size 8 \
  --learning_rate 1e-5 --data_path ../dataset/sft_gucheng_cot.jsonl \
  --from_weight full_sft_gucheng --max_seq_len 512 --num_workers 0
```
