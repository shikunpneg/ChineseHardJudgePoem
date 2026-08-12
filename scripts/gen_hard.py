# -*- coding: utf-8 -*-
"""hard 难样本生成：用指定微调模型批量模仿创作诗歌，并计算与真实诗歌的相似度。

用法:
  python gen_hard.py --model models\\Qwen2.5-3B-GuCheng --group GuCheng --out hard_gen_GuCheng.jsonl
  python gen_hard.py --model models\\Qwen2.5-3B-Haizi --group Haizi --out hard_gen_Haizi.jsonl
  python gen_hard.py --model models\\Qwen2.5-3B-Haizi-CN --group Haizi-CN --out hard_gen_Haizi-CN.jsonl
  python gen_hard.py --model models\\Qwen2.5-3B-LiBai-Final --group LiBai --out hard_gen_LiBai.jsonl

相似度指标：
  - sim_jaccard : 字符 2-gram 集合 Jaccard 相似度（vs 同标题真实诗）
  - sim_cosine  : 字符 2-gram 词频向量余弦相似度（vs 同标题真实诗）
"""
import argparse
import json
import math
import os
import random
import re
import sys
import time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(HERE, "..", "llama_data", "hard_input_5000.jsonl")
OUT_DIR = os.path.join(HERE, "..", "hard_data")
DATA_DIR = os.path.join(HERE, "..", "llama_data")

BATCH = 6
MAX_NEW = 200

# 各模型对应的风格真实诗池（用于计算风格保真度 sim_pool）
STYLE_POOLS = {
    "GuCheng": "gucheng_train.jsonl",
    "Haizi": "haizi_train.jsonl",
    "Haizi-CN": "haizi_cn_train.jsonl",
    "LiBai": "libai_train.jsonl",
}
POOL_SAMPLE = 20  # 每条生成 vs 风格池随机采样条数
SEED = 20260812


def patch_transformers_tokenizer_bug():
    import transformers.tokenization_utils_base as tub
    _orig = tub.PreTrainedTokenizerBase._set_model_specific_special_tokens

    def _patched(self, special_tokens=None):
        if special_tokens is None:
            return
        if isinstance(special_tokens, dict):
            _orig(self, special_tokens)

    tub.PreTrainedTokenizerBase._set_model_specific_special_tokens = _patched


# ---------- 相似度 ----------
def bigrams(text):
    text = re.sub(r"\s+", "", text)
    return [text[i:i + 2] for i in range(len(text) - 1)]


def jaccard(a, b):
    A, B = set(bigrams(a)), set(bigrams(b))
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def cosine(a, b):
    from collections import Counter
    va, vb = Counter(bigrams(a)), Counter(bigrams(b))
    if not va or not vb:
        return 0.0
    dot = sum(c * vb.get(g, 0) for g, c in va.items())
    na = math.sqrt(sum(c * c for c in va.values()))
    nb = math.sqrt(sum(c * c for c in vb.values()))
    if na * nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------- 悬空字修剪 ----------
END_CHARS = "。！？；…」』）】"


def trim_incomplete(text):
    t = text.strip()
    while t and t[-1] in "，、：,":
        t = t[:-1].strip()
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="模型路径或 HF 仓库")
    ap.add_argument("--group", required=True, help="组名: GuCheng / Haizi / Haizi-CN / LiBai")
    ap.add_argument("--out", default=None, help="输出文件名（默认 hard_gen_<group>.jsonl）")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_new", type=int, default=MAX_NEW)
    ap.add_argument("--n", type=int, default=0, help="只跑前 n 条（调试用）")
    args = ap.parse_args()

    out_name = args.out or f"hard_gen_{args.group}.jsonl"
    out_path = os.path.join(OUT_DIR, out_name)
    os.makedirs(OUT_DIR, exist_ok=True)

    # 读取该组输入
    rows = []
    with open(INPUT, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["meta"]["model"] == args.group:
                rows.append(rec)
    if args.n:
        rows = rows[: args.n]
    print(f"[{args.group}] {len(rows)} prompts -> {out_path}", flush=True)
    assert rows, f"group {args.group} 无数据"

    patch_transformers_tokenizer_bug()
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto",
        quantization_config=quant_cfg,
    )
    model.eval()

    gen_cfg = dict(
        max_new_tokens=args.max_new,
        temperature=args.temperature,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.eos_token_id,
    )

    # 加载该模型风格真实诗池（计算风格保真度）
    style_recs = []
    if args.group in STYLE_POOLS:
        with open(os.path.join(DATA_DIR, STYLE_POOLS[args.group]), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                gpt = next((m["value"] for m in rec.get("conversations", []) if m.get("from") == "gpt"), "")
                if gpt.strip():
                    style_recs.append(gpt)
    rng = random.Random(SEED)
    print(f"[{args.group}] style pool: {len(style_recs)} poems", flush=True)

    # 预编码（按批左填充）
    def encode(recs):
        msgs_list = []
        for rec in recs:
            conv = rec["conversations"]
            msgs = []
            for m in conv:
                if m["from"] == "system":
                    msgs.append({"role": "system", "content": m["value"]})
                elif m["from"] == "human":
                    msgs.append({"role": "user", "content": m["value"]})
            msgs_list.append(msgs)
        enc = tokenizer.apply_chat_template(
            msgs_list, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", padding=True)
        input_ids = enc["input_ids"] if hasattr(enc, "keys") else enc
        attn = enc.get("attention_mask") if hasattr(enc, "get") else None
        if attn is not None:
            attn = attn.to(model.device)
        return input_ids.to(model.device), attn

    t0 = time.time()
    results = []
    sims = []
    with open(out_path, "w", encoding="utf-8") as fo:
        for i in range(0, len(rows), BATCH):
            chunk = rows[i:i + BATCH]
            input_ids, attn = encode(chunk)
            with torch.no_grad():
                out = model.generate(inputs=input_ids, attention_mask=attn, **gen_cfg)
            gen_ids = out[:, input_ids.shape[1]:]
            texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            for rec, text in zip(chunk, texts):
                gen = trim_incomplete(text.strip())
                meta = rec["meta"]
                ref = next(m["value"] for m in rec["conversations"] if m["from"] == "gpt")
                sj = jaccard(gen, ref)
                sc = cosine(gen, ref)
                # 风格池平均相似度（vs 随机采样 POOL_SAMPLE 首真实诗）
                if style_recs:
                    pool = rng.sample(style_recs, min(POOL_SAMPLE, len(style_recs)))
                    sp = sum(jaccard(gen, p) for p in pool) / len(pool)
                else:
                    sp = 0.0
                sims.append((sj, sc, sp))
                rec_out = {
                    "model": meta["model"],
                    "title": meta["title"],
                    "genre": meta["genre"],
                    "prompt": next(m["value"] for m in rec["conversations"] if m["from"] == "human"),
                    "generated": gen,
                    "real_text": ref,
                    "sim_jaccard": round(sj, 4),
                    "sim_cosine": round(sc, 4),
                    "sim_pool": round(sp, 4),
                }
                fo.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
                results.append(rec_out)
            fo.flush()
            if (i // BATCH + 1) % 10 == 0:
                n = len(results)
                avg_j = sum(s[0] for s in sims) / max(len(sims), 1)
                avg_p = sum(s[2] for s in sims) / max(len(sims), 1)
                print(f"  {n}/{len(rows)} avg_jaccard={avg_j:.3f} avg_pool={avg_p:.3f} elapsed={time.time()-t0:.0f}s", flush=True)

    # 统计
    n = len(sims)
    jacs = [s[0] for s in sims]
    coss = [s[1] for s in sims]
    pools = [s[2] for s in sims]
    stats = {
        "group": args.group,
        "count": n,
        "jaccard_mean": round(sum(jacs) / n, 4),
        "jaccard_median": round(sorted(jacs)[n // 2], 4),
        "jaccard_max": round(max(jacs), 4),
        "jaccard_over_0_4": round(sum(1 for x in jacs if x > 0.4) / n, 4),
        "jaccard_over_0_6": round(sum(1 for x in jacs if x > 0.6) / n, 4),
        "cosine_mean": round(sum(coss) / n, 4),
        "pool_mean": round(sum(pools) / n, 4),
        "pool_median": round(sorted(pools)[n // 2], 4),
        "elapsed_sec": int(time.time() - t0),
    }
    stat_path = os.path.join(OUT_DIR, f"hard_stats_{args.group}.json")
    with open(stat_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[{args.group}] DONE {n} rows, avg_jaccard={stats['jaccard_mean']}, avg_pool={stats['pool_mean']}, {stats['elapsed_sec']}s")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
