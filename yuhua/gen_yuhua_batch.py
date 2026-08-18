# -*- coding: utf-8 -*-
"""余华风格模型批量生成脚本（AR CoT 模式）
用法（从 minimind 根目录）:
  python scripts/gen_yuhua_batch.py --weight full_sft_yuhua_cot --n 100 --out ../dataset/../out/yuhua_samples_v1.jsonl
"""
import sys, os, re, json, math, random, argparse, time
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from transformers import AutoTokenizer

# 余华风格题目池（来自真实作品主题，非虚构）
PROMPT_POOL = [
    "请用余华的风格，写一段关于卖血的小镇故事。",
    "请用余华的风格，写一段关于饥饿的记忆。",
    "请用余华的风格，写一段父子之间的沉默。",
    "请用余华的风格，写一段葬礼上的描写。",
    "请用余华的风格，写一段关于贫穷的日子。",
    "请用余华的风格，写一段兄弟重逢的场景。",
    "请用余华的风格，写一段村庄里的一场火灾。",
    "请用余华的风格，写一段关于死亡的叙述。",
    "请用余华的风格，写一段医院里的等待。",
    "请用余华的风格，写一段关于土地和劳作。",
    "请用余华的风格，写一段母亲送别儿子。",
    "请用余华的风格，写一段雨中的街道。",
    "请用余华的风格，写一段关于偷窃的往事。",
    "请用余华的风格，写一段老人讲述过去。",
    "请用余华的风格，写一段关于牛的描写。",
]

SYSTEM_YUHUA = ("你是一位深谙余华小说风格的写作者。余华的写作特点是：白描、克制、不直接抒情，"
                "用具体的物件和动作呈现情感；写底层人物在苦难中的生存；语气冷峻平静，"
                "偶尔有荒诞与反讽；短句，重复，让事实自己说话。")

def load_model(weight, device="cuda:0"):
    lm_config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=False)
    model = MiniMindForCausalLM(lm_config)
    path = f"{ROOT}/out/{weight}_768.pth"
    weights = torch.load(path, map_location=device)
    model.load_state_dict(weights, strict=False)
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(f"{ROOT}/model")
    return model, tokenizer, lm_config

def qualify(t):
    if len(t) < 40 or len(t) > 700:
        return False
    if "\ufffd" in t:
        return False
    if re.search(r"[A-Za-z]", t):
        return False
    if re.search(r"[0-9]", t):
        return False
    if re.search(r"\{\"|\}|```", t):
        return False
    # 重复率
    seq = t[:400]
    for k in (8, 12, 16, 24):
        cnt = len(re.findall(f"(.{{{k}}})\\1", seq))
        if cnt >= 2:
            return False
    return True

@torch.no_grad()
def gen(model, tokenizer, lm_config, prompt, device="cuda:0", max_new=220, temperature=1.0,
        top_p=0.9, top_k=50, repetition_penalty=1.1):
    msgs = [{"role": "system", "content": SYSTEM_YUHUA}, {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    out = model.generate(input_ids=input_ids,
                         attention_mask=torch.ones_like(input_ids),
                         max_new_tokens=max_new,
                         temperature=temperature,
                         top_p=top_p, top_k=top_k,
                         repetition_penalty=repetition_penalty,
                         do_sample=True)
    return tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", default="full_sft_gucheng_cot")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_new", type=int, default=220)
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model, tokenizer, lm_config = load_model(args.weight, args.device)
    out_path = args.out or f"{ROOT}/out/yuhua_samples_{args.weight}.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    generated = 0
    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(args.n):
            prompt = PROMPT_POOL[i % len(PROMPT_POOL)]
            for _ in range(40):
                txt = gen(model, tokenizer, lm_config, prompt, args.device, args.max_new,
                          args.temperature)
                if qualify(txt):
                    break
                txt = None
            if txt:
                generated += 1
                rec = {"prompt": prompt, "text": txt, "model": args.weight, "version": "v1"}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[{i+1}/{args.n}] ok len={len(txt)} ({time.time()-t0:.0f}s)")
            else:
                print(f"[{i+1}/{args.n}] FAIL")
    print(f"done: {generated}/{args.n} -> {out_path}")

if __name__ == "__main__":
    main()
