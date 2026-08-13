# -*- coding: utf-8 -*-
"""计算相似度基线，用于解读 hard 数据集指标量级（论文 4.3 节）。

基线 1 random_vs_pool : 随机中文字符文本 vs 真实诗池的平均 Jaccard（下界）
基线 2 real_vs_pool    : 真实诗 vs 同池其他真实诗的平均 Jaccard（自相似上界参考）
基线 3 base_model      : 待 GPU 空闲后由 Qwen2.5-3B-Instruct 基座生成（另跑）
"""
import json
import os
import random
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "llama_data")
OUT = os.path.join(HERE, "..", "hard_data", "baseline_random.json")

FILES = ["gucheng_train.jsonl", "haizi_train.jsonl", "haizi_cn_train.jsonl", "libai_train.jsonl"]
SEED = 20260812
N_SAMP = 300  # 采样条数


def bigrams(text):
    text = re.sub(r"\s+", "", text)
    return [text[i:i + 2] for i in range(len(text) - 1)]


def jaccard(a, b):
    A, B = set(bigrams(a)), set(bigrams(b))
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def main():
    rng = random.Random(SEED)
    pool = []
    for fname in FILES:
        with open(os.path.join(DATA_DIR, fname), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                gpt = next((m["value"] for m in rec.get("conversations", []) if m.get("from") == "gpt"), "")
                if gpt.strip():
                    pool.append(gpt)
    rng.shuffle(pool)
    print(f"poem pool: {len(pool)}")

    # 基线 1：随机中文字符文本
    CJK = list("天地玄黄宇宙洪荒日月盈昃辰宿列张寒来暑往秋收冬藏诗酒花月山水风云江河湖海草木春秋心骨神情魂梦歌哭日月星")
    rand_sims = []
    for _ in range(N_SAMP):
        L = rng.randint(20, 120)
        rand_txt = "".join(rng.choice(CJK) for _ in range(L))
        refs = rng.sample(pool, 20)
        rand_sims.append(sum(jaccard(rand_txt, r) for r in refs) / len(refs))
    r1 = {"mean": sum(rand_sims) / len(rand_sims),
          "median": sorted(rand_sims)[len(rand_sims) // 2]}

    # 基线 2：真实诗 vs 其他真实诗
    real_sims = []
    for _ in range(N_SAMP):
        t = rng.choice(pool)
        refs = rng.sample(pool, 20)
        real_sims.append(sum(jaccard(t, r) for r in refs) / len(refs))
    r2 = {"mean": sum(real_sims) / len(real_sims),
          "median": sorted(real_sims)[len(real_sims) // 2]}

    out = {
        "seed": SEED,
        "n_samples": N_SAMP,
        "random_vs_pool": r1,
        "real_vs_pool": r2,
        "note": "random_vs_pool 为下界；real_vs_pool 为真实诗与同池他诗的自相似参考；生成模型 sim_pool 应落于二者之间",
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
