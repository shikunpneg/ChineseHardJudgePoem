# -*- coding: utf-8 -*-
"""
从 hard_dataset_5000.jsonl 中剔除"背诵样本"，并筛选"文本相似度接近"的无标签待标注子集。

判定规则：
- 背诵样本(recited)：sim_jaccard >= RECITE_JACCARD（与同标题真实诗近乎逐字相同）
- 文本相似度接近(near)：sim_jaccard >= NEAR_JACCARD 且非背诵样本
- 输出均不含任何答案/标签，供人工标注（判断生成文本与真实文本的关系/是否抄袭等）

输出：
  data/recited_removed.jsonl   被剔除的背诵样本（清单，附原因）
  data/to_annotate_near.jsonl  文本相似度接近的无标签待标注样本
  data/to_annotate_report.md   剔除与筛选统计报告
"""
import json
import os
import statistics

SRC = os.path.join(os.path.dirname(__file__), "..", "hard_data", "hard_dataset_5000.jsonl")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "hard_data")
RECITE_JACCARD = 0.9
NEAR_JACCARD = 0.1
NEAR_COSINE = 0.3

def norm(s: str) -> str:
    """去空白/标点归一化，用于背诵检测"""
    return "".join(ch for ch in s if not ch.isspace() and ch not in "，。！？、；：,.!?;:()（）")

def main():
    with open(SRC, encoding="utf-8") as f:
        recs = [json.loads(l) for l in f if l.strip()]
    print(f"输入总样本: {len(recs)}")

    recited, kept = [], []
    for r in recs:
        ng, nr = norm(r["generated"]), norm(r["real_text"])
        if r["sim_jaccard"] >= RECITE_JACCARD or (ng and ng == nr):
            r["remove_reason"] = "背诵样本(sim_jaccard=%.4f)" % r["sim_jaccard"]
            recited.append(r)
        else:
            kept.append(r)

    near = [r for r in kept if r["sim_jaccard"] >= NEAR_JACCARD or r["sim_cosine"] >= NEAR_COSINE]
    # 输出字段：仅保留展示信息，不携带任何标签/答案
    def strip(r):
        return {
            "id": r.get("id", ""),
            "model": r["model"],
            "title": r["title"],
            "genre": r["genre"],
            "prompt": r["prompt"],
            "generated": r["generated"],
            "real_text": r["real_text"],
            "sim_jaccard": round(r["sim_jaccard"], 4),
            "sim_cosine": round(r["sim_cosine"], 4),
        }

    recited_out = [{**strip(r), "remove_reason": r["remove_reason"]} for r in recited]
    near_out = [strip(r) for r in near]

    os.makedirs(OUT_DIR, exist_ok=True)
    p_rec = os.path.join(OUT_DIR, "recited_removed.jsonl")
    p_near = os.path.join(OUT_DIR, "to_annotate_near.jsonl")
    with open(p_rec, "w", encoding="utf-8") as f:
        for r in recited_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(p_near, "w", encoding="utf-8") as f:
        for r in near_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 报告
    def grp(rs, key):
        d = {}
        for r in rs:
            d[r[key]] = d.get(r[key], 0) + 1
        return dict(sorted(d.items()))

    lines = [
        "# Hard 难样本：背诵样本剔除与待标注筛选报告",
        "",
        "## 1. 判定规则",
        "",
        f"- 背诵样本：`sim_jaccard >= {RECITE_JACCARD}` 或归一化后文本与真实诗完全相同（模型逐字复刻）",
        f"- 文本相似度接近：`sim_jaccard >= {NEAR_JACCARD}` 或 `sim_cosine >= {NEAR_COSINE}`，且非背诵样本",
        "- 待标注数据不含任何答案/标签，需人工判断。",
        "",
        "## 2. 剔除的背诵样本",
        "",
        f"- 数量：**{len(recited)}**（占比 {len(recited)/len(recs)*100:.2f}%）",
        f"- 按模型：{grp(recited, 'model')}",
        "",
        "| model | title | sim_jaccard | sim_cosine |",
        "|-------|-------|-------------|------------|",
    ]
    for r in sorted(recited, key=lambda x: -x["sim_jaccard"]):
        lines.append(f"| {r['model']} | {r['title']} | {r['sim_jaccard']:.4f} | {r['sim_cosine']:.4f} |")
    lines += [
        "",
        "## 3. 文本相似度接近的待标注样本",
        "",
        f"- 数量：**{len(near)}**（剔除背诵后剩余 {len(kept)} 条中的保留子集）",
        f"- 按模型：{grp(near, 'model')}",
        f"- sim_jaccard 均值：{statistics.mean(r['sim_jaccard'] for r in near):.4f}，中位数：{statistics.median(r['sim_jaccard'] for r in near):.4f}",
        f"- sim_cosine 均值：{statistics.mean(r['sim_cosine'] for r in near):.4f}",
        "",
        "## 4. 说明",
        "",
        "- 背诵样本因答案明确（逐字抄袭真实诗）不宜放入标注网站，已剔除；",
        "- 保留的均为与真实诗歌文本相似度接近、具有判别难度的样本；",
        "- 标注任务：由人工判断生成文本（generated）与真实文本（real_text）的关系（如：是否同一首诗、是否抄袭、相似度等级等），输出无预置答案。",
    ]
    p_rep = os.path.join(OUT_DIR, "to_annotate_report.md")
    with open(p_rep, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"背诵样本: {len(recited)} -> {p_rec}")
    print(f"待标注接近样本: {len(near)} -> {p_near}")
    print(f"报告: {p_rep}")

if __name__ == "__main__":
    main()
