# -*- coding: utf-8 -*-
"""合并四组 hard 生成结果 -> hard_dataset_5000.jsonl + 相似度统计报告（Markdown）。"""
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
HARD_DIR = os.path.join(HERE, "..", "hard_data")
GEN_FILES = [
    "hard_gen_GuCheng.jsonl",
    "hard_gen_Haizi.jsonl",
    "hard_gen_Haizi-CN.jsonl",
    "hard_gen_LiBai.jsonl",
]
OUT_DS = os.path.join(HARD_DIR, "hard_dataset_5000.jsonl")
OUT_REPORT = os.path.join(HARD_DIR, "hard_dataset_similarity_report.md")


def load_gen(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stats(rows):
    jacs = [r["sim_jaccard"] for r in rows]
    coss = [r["sim_cosine"] for r in rows]
    jacs_s = sorted(jacs)
    n = len(jacs)
    return {
        "n": n,
        "mean": sum(jacs) / n,
        "median": jacs_s[n // 2],
        "max": max(jacs),
        "gt04": sum(1 for x in jacs if x > 0.4) / n,
        "gt06": sum(1 for x in jacs if x > 0.6) / n,
        "cos": sum(coss) / n,
    }


def main():
    all_rows = []
    for fname in GEN_FILES:
        rows = load_gen(os.path.join(HARD_DIR, fname))
        print(f"{fname}: {len(rows)}")
        all_rows.extend(rows)

    assert len(all_rows) >= 4000, f"生成数量不足: {len(all_rows)}"
    all_rows = all_rows[:5000]

    with open(OUT_DS, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_model = Counter(r["model"] for r in all_rows)
    groups = {}
    for r in all_rows:
        groups.setdefault(r["model"], []).append(r)

    lines = []
    lines.append("# Hard 难样本数据集：相似度统计报告\n")
    lines.append("## 1. 数据集概况\n")
    lines.append(f"- 总样本数：**{len(all_rows)}**")
    lines.append("- 生成模型：四个 QLoRA 微调模型（GuCheng / Haizi / Haizi-CN / LiBai）")
    lines.append(f"- 各模型样本数：{dict(by_model)}")
    lines.append("- 生成配置：4bit 量化加载、temperature=1.0、top_p=0.9、repetition_penalty=1.05")
    lines.append("- 相似度指标：字符 2-gram 集合 Jaccard 相似度 与 字符 2-gram 词频余弦相似度（对同标题真实诗）")
    lines.append("")
    lines.append("## 2. 按模型相似度统计\n")
    lines.append("| 模型 | 样本数 | Jaccard 均值 | Jaccard 中位数 | Jaccard 最大 | Jaccard>0.4 占比 | Jaccard>0.6 占比 | 余弦均值 |")
    lines.append("|------|--------|-------------|----------------|--------------|------------------|------------------|----------|")

    for model in ["GuCheng", "Haizi", "Haizi-CN", "LiBai"]:
        s = stats(groups[model])
        lines.append(f"| {model} | {s['n']} | {s['mean']:.4f} | {s['median']:.4f} | {s['max']:.4f} "
                     f"| {s['gt04']:.2%} | {s['gt06']:.2%} | {s['cos']:.4f} |")

    s = stats(all_rows)
    lines.append(f"| **全部** | {s['n']} | **{s['mean']:.4f}** | **{s['median']:.4f}** | **{s['max']:.4f}** "
                 f"| **{s['gt04']:.2%}** | **{s['gt06']:.2%}** | **{s['cos']:.4f}** |")
    lines.append("")
    lines.append("## 3. 结论\n")
    lines.append("- 生成诗歌与真实诗歌的平均字符 2-gram Jaccard 相似度（对同标题）反映了模型对原风格的**模仿强度**；")
    lines.append("- 相似度越高，生成的诗歌越难以与真实诗歌区分，即样本越“难”；")
    lines.append("- 该数据集可作为诗歌真伪判别任务（poetry-judge）的难样本测试集，或用于评估生成模型的风格保真度。")

    report = "\n".join(lines)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nWROTE {OUT_DS}")
    print(f"WROTE {OUT_REPORT}")
    print(report)


if __name__ == "__main__":
    main()
