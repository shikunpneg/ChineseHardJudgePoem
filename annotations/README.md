# 人工标注导出（v2 标注系统，2026-08-15）

数据源：[eval-annotation/](file:///e:/生成诗歌/eval-annotation/) 标注平台 v2，导出时间 2026-08-15 15:51–15:53（Beijing time）。
标注样本池：本项目 `data/to_annotate_near.jsonl`（基于 v2 难样本相似度接近但**非背诵**的近邻样本，共 194 条），由标注员进行二元（是否诗）+ 4 级质量分（quality_grade）双轴评判。

## 文件

| 文件 | 行数 | 说明 |
|---|---|---|
| `annotations_export_hk_20260815.csv` | 1764 标注 | 全量原始导出（包含已提交和未提交的标注），列：username, user_role, sample_id, title, author, truth_genre, source_type, origin, is_poetry, quality_grade, updated_at |
| `annotations_hk_graded_20260815.csv` | 215 标注 | 已提交且 **quality_grade 非空**的子集（即真正完成打分的高质量标注） |

## 字段说明

- `username` / `user_role`：标注员 ID 与角色（全部为 `annotator`）
- `sample_id`：标注平台样本主键，对应 `data/to_annotate_near.jsonl` 的样本 ID
- `title` / `author` / `truth_genre` / `source_type`：样本诗作元数据（诗/非诗、古代/现代、AI/人类）
- `origin`：样本来源标识（本批均为空 → 由平台默认映射）
- `is_poetry`：二元标注，标注员判定该样本是否为诗（`true` / `false`）
- `quality_grade`：质量分，1（差）– 4（优），空表示该条未完成打分
- `updated_at`：最后更新时间（UTC）

## 已知差异

- `annotations_export_hk_20260815.csv` 行数 = `annotations_hk.csv`（08-14 版 1764 行）+ 1 行（08-15 新增），可视为递增补丁。
- 旧版标注 `annotations_export.csv` (1189 行)、`annotations_export2.csv` (143 行)、`annotations_export6.csv` (424 行) 来自 v1 标注平台（不同样本池），与本次 v2 标注不混用。

## 引用方式

```python
import pandas as pd
df = pd.read_csv("annotations/annotations_hk_graded_20260815.csv")
print(df.groupby(["author", "is_poetry", "quality_grade"]).size())
```