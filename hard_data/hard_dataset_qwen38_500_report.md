# Qwen3.8-27B Hard 难样本数据集（500 条）：统计报告

## 1. 概况

- 总样本数：**500**
- 生成模型：**Qwen3.8-27B**（最新 Qwen，关闭 thinking 模式，4bit 量化推理）
- 各风格样本数：{'Haizi': 144, 'GuCheng': 111, 'Haizi-CN': 114, 'LiBai': 131}
- 生成配置：temperature=1.0、top_p=0.9、repetition_penalty=1.05、max_new_tokens=200、seed=20260812
- 相似度指标：字符 2-gram 集合 Jaccard / 词频余弦（对同标题真实诗）+ 风格池平均 Jaccard (sim_pool)

## 2. 总体统计

| 指标 | 值 |
|------|-----|
| n | 500 |
| jaccard_mean | 0.0168 |
| jaccard_median | 0.0136 |
| jaccard_max | 0.4798 |
| jaccard_over_0_4 | 0.002 |
| jaccard_over_0_6 | 0.0 |
| cosine_mean | 0.0641 |
| pool_mean | 0.0098 |
| pool_median | 0.0111 |

## 3. 按风格统计

| 模型 | 样本数 | Jaccard 均值 | 余弦均值 | sim_pool 均值 |
|------|--------|-------------|----------|---------------|
| GuCheng | 111 | 0.0181 | 0.0693 | 0.0116 |
| Haizi | 144 | 0.0182 | 0.0741 | 0.0125 |
| Haizi-CN | 114 | 0.0232 | 0.0779 | 0.012 |
| LiBai | 131 | 0.0086 | 0.0365 | 0.0034 |

## 4. 结论

- Qwen3.8-27B 生成的诗歌在字符 2-gram 层面与真实诗歌保持结构相似性，可作为诗歌真伪判别任务的困难负样本；
- 与 Qwen2.5-3B 微调模型的对比见实验结论。