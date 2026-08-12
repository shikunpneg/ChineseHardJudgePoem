# -*- coding: utf-8 -*-
"""构建 5000 条 hard 难样本生成输入（4 组 x 1250 条，对应四个微调模型）。

每个模型的生成提示使用其训练时的 system 风格提示 + 从统一标题池采样的标题。
每条记录保留同标题真实诗歌作为参考文本（ref），供后续相似度计算。
"""
import json
import random
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "llama_data")
OUT = os.path.join(DATA_DIR, "hard_input_5000.jsonl")

TRAIN_SETS = {
    "GuCheng": "gucheng_train.jsonl",
    "Haizi": "haizi_train.jsonl",
    "Haizi-CN": "haizi_cn_train.jsonl",
    "LiBai": "libai_train.jsonl",
}

GROUP_SIZE = 1250  # 每模型生成数
SEED = 20260812


def load_train(path):
    """返回 (titles: list[dict], system: str, first_human: str)"""
    recs = []
    system = None
    first_human = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            conv = rec.get("conversations", [])
            if not conv:
                continue
            if conv[0].get("from") == "system":
                system = conv[0]["value"]
            human = next((m["value"] for m in conv if m.get("from") == "human"), "")
            gpt = next((m["value"] for m in conv if m.get("from") == "gpt"), "")
            if not first_human:
                first_human = human
            meta = rec.get("meta", {}) or {}
            recs.append({
                "title": meta.get("title", ""),
                "genre": meta.get("genre", ""),
                "author": meta.get("author", ""),
                "text": gpt,
                "human": human,
            })
    return recs, system, first_human


def main():
    rng = random.Random(SEED)
    data = {}
    title_pool = []
    seen = set()
    for model, fname in TRAIN_SETS.items():
        path = os.path.join(DATA_DIR, fname)
        recs, system, first_human = load_train(path)
        data[model] = {"recs": recs, "system": system, "human": first_human}
        for r in recs:
            t = r["title"]
            if t and t not in seen:
                seen.add(t)
                title_pool.append(r)
        print(f"[{model}] {len(recs)} records, system={system[:20] if system else None}...")

    print(f"title pool: {len(title_pool)} unique titles")
    assert len(title_pool) >= 1

    # human 提示模板（沿用各模型训练时的提示风格）
    def human_prompt(model, title):
        if model == "LiBai":
            return f"你是诗人李白，请以《{title}》为题，写一首你觉得好的诗歌。"
        return f"请以《{title}》为题，创作一首现代诗。"

    rows = []
    for model in TRAIN_SETS:
        system = data[model]["system"]
        for _ in range(GROUP_SIZE):
            ref = rng.choice(title_pool)
            rows.append({
                "model": model,
                "system": system,
                "human": human_prompt(model, ref["title"]),
                "title": ref["title"],
                "genre": ref["genre"] or "现代诗",
                "ref_text": ref["text"],
                "ref_author": ref.get("author", ""),
            })

    rng.shuffle(rows)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            rec = {
                "conversations": [
                    {"from": "system", "value": r["system"]},
                    {"from": "human", "value": r["human"]},
                    {"from": "gpt", "value": r["ref_text"]},
                ],
                "meta": {
                    "model": r["model"],
                    "title": r["title"],
                    "genre": r["genre"],
                    "ref_author": r["ref_author"],
                },
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"WROTE {len(rows)} rows -> {OUT}")

    from collections import Counter
    print("by model:", dict(Counter(r["model"] for r in rows)))


if __name__ == "__main__":
    main()
