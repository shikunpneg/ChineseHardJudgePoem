# -*- coding: utf-8 -*-
"""余华场景三模型对比生成：AR / dLM / Linear 用同一批长文提示，评测连贯性
用法（从 minimind 根目录运行）:
  python scripts/gen_yuhua_compare.py --model ar     --weight full_sft_yuhua_cot      --n 30
  python scripts/gen_yuhua_compare.py --model dllm   --weight dllm_sft_yuhua          --n 30
  python scripts/gen_yuhua_compare.py --model linear --weight full_sft_linear_yuhua_cot --n 30
"""
import sys, os, math, json, argparse, random, datetime, torch, re
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

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

MODEL_NAMES = {'ar': 'MiniMind-YuHua-AR-CoT', 'dllm': 'MiniMind-YuHua-dLM', 'linear': 'MiniMind-YuHua-Linear'}


def load_model(model_name, device, weight_name):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('model')
    if model_name == 'dllm':
        from model.model_minimind_dllm import MiniMindDLLMConfig, MiniMindForMaskedDiffusion
        config = MiniMindDLLMConfig(hidden_size=768)
        model = MiniMindForMaskedDiffusion(config)
        model.load_state_dict(torch.load(f'out/{weight_name}_768.pth', map_location=device), strict=False)
        model.half().eval().to(device)
        return model, tokenizer, 'dllm'
    if model_name == 'linear':
        import importlib
        sys.modules['model.model_minimind'] = importlib.import_module('model.model_minimind_linear')
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    config = MiniMindConfig(hidden_size=768)
    model = MiniMindForCausalLM(config)
    model.load_state_dict(torch.load(f'out/{weight_name}_768.pth', map_location=device), strict=False)
    model.to(device).eval()
    return model, tokenizer, 'ar'


def gen_ar(model, tokenizer, prompt, device, temperature, max_new_tokens=240):
    messages = [{'role': 'system', 'content': SYSTEM_YUHUA}, {'role': 'user', 'content': prompt}]
    p = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(p, return_tensors='pt').input_ids.to(device)
    with torch.inference_mode():
        out = model.generate(input_ids=input_ids,
                             attention_mask=torch.ones_like(input_ids),
                             max_new_tokens=max_new_tokens,
                             temperature=temperature,
                             top_p=0.9, top_k=50,
                             repetition_penalty=1.1,
                             do_sample=True)
    return tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()


def gen_dllm(model, tokenizer, prompt, device, temperature, max_new_tokens=320):
    messages = [{'role': 'system', 'content': SYSTEM_YUHUA}, {'role': 'user', 'content': prompt}]
    p = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(p, return_tensors='pt', truncation=True).input_ids.to(device)
    prompt_len = input_ids.shape[1]
    mask_id, eos_id = model.config.mask_token_id, tokenizer.eos_token_id
    block_size, steps = 64, 28
    num_blocks = math.ceil(max_new_tokens / block_size)
    T = prompt_len + max_new_tokens
    x = torch.full((1, T), eos_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids
    x[:, prompt_len:] = mask_id
    from model.model_minimind_dllm import add_gumbel_noise
    with torch.inference_mode():
        for b in range(num_blocks):
            block_end = min(prompt_len + (b + 1) * block_size, T)
            for step in range(steps):
                mask_index = (x == mask_id)
                mask_count = mask_index[:, :block_end].sum(-1).min().item()
                if mask_count == 0:
                    break
                n_unmask = max(1, round(mask_count / (steps - step)))
                logits = model(input_ids=x).logits
                logits[logits < torch.topk(logits, 50, dim=-1)[0][..., -1:]] = -float('inf')
                x0 = torch.argmax(add_gumbel_noise(logits, temperature), dim=-1)
                p_prob = F.softmax(logits.float(), dim=-1)
                x0_p = torch.gather(p_prob, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                x0_p[:, block_end:] = -float('inf')
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, torch.tensor(-float('inf'), device=device))
                _, idx = torch.topk(confidence[0], k=min(n_unmask, int(mask_count)))
                x[0, idx] = x0[0, idx]
            if eos_id and (x[:, prompt_len:] == eos_id).any():
                break
    gen_ids = x[0, prompt_len:].tolist()
    if eos_id in gen_ids:
        gen_ids = gen_ids[:gen_ids.index(eos_id)]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def qualify(text):
    t = text.strip()
    if len(t) < 40:
        return False
    if len(t) > 900:
        return False
    if '\ufffd' in t or '\u0000' in t:
        return False
    if any('A' <= c <= 'z' for c in t):
        return False
    if any('0' <= c <= '9' for c in t):
        return False
    if '{"' in t or '}' in t or '```' in t:
        return False
    flat = t.replace('\n', '')
    if len(flat) >= 8 and flat.count(flat[:8]) > 3:
        return False
    for k in (12, 16, 24):
        cnt = len(re.findall(f'(.{{{k}}})\\1', flat[:600]))
        if cnt >= 2:
            return False
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, required=True, choices=['ar', 'dllm', 'linear'])
    p.add_argument('--weight', type=str, required=True)
    p.add_argument('--n', type=int, default=30)
    p.add_argument('--version', type=str, default='v1')
    p.add_argument('--out', type=str, default=None)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max_new', type=int, default=240)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    model, tokenizer, kind = load_model(args.model, args.device, args.weight)
    out_path = args.out or f'{ROOT}/out/yuhua_cmp_{args.model}_{args.version}.jsonl'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    report = {'model': MODEL_NAMES[args.model], 'weight': args.weight, 'version': args.version,
              'target_n': args.n, 'generated': 0, 'filtered': 0}
    samples = []
    idx = 0
    t0 = datetime.datetime.now()
    while len(samples) < args.n:
        prompt = PROMPT_POOL[idx % len(PROMPT_POOL)]
        idx += 1
        temp = {'ar': 1.0, 'dllm': 0.7, 'linear': 0.6}[args.model]
        report['generated'] += 1
        try:
            if kind == 'dllm':
                text = gen_dllm(model, tokenizer, prompt, args.device, temp, args.max_new)
            else:
                text = gen_ar(model, tokenizer, prompt, args.device, temp, args.max_new)
        except Exception as e:
            print(f'  [异常] {e}')
            continue
        if not qualify(text):
            report['filtered'] += 1
            print(f'  [{len(samples)+1}/{args.n}] 过滤 len={len(text)}: {text[:24]!r}')
            continue
        rec = {
            'author': '余华风格', 'model': MODEL_NAMES[args.model], 'version': args.version,
            'mode': 'chat', 'prompt': prompt, 'temperature': temp,
            'content': text,
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        samples.append(rec)
        with open(out_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print(f'  [{len(samples)}/{args.n}] len={len(text)} 耗时={(datetime.datetime.now()-t0).seconds}s')
    with open(f'{ROOT}/out/report_yuhua_cmp_{args.model}_{args.version}.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f'\n[完成] {out_path}  合格 {len(samples)} 条，共生成 {report["generated"]}，过滤 {report["filtered"]}')


if __name__ == '__main__':
    main()
