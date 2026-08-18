# -*- coding: utf-8 -*-
"""把 pretrain_yuhua_pure.jsonl 转成 conversations 格式（pretrain_yuhua_pure_chat.jsonl），
给 dLM/Linear 等只能吃 SFTDataset 的脚本用（单轮对话：user 给前 80 字，assistant 续完整段）
"""
import json, os, random
random.seed(42)
SRC = r"E:\生成诗歌\minimind\dataset\pretrain_yuhua_pure.jsonl"
OUT = r"E:\生成诗歌\minimind\dataset\pretrain_yuhua_pure_chat.jsonl"
SYSTEM = ("你是一位深谙余华小说风格的写作者。余华的写作特点是：白描、克制、不直接抒情，"
          "用具体的物件和动作呈现情感；写底层人物在苦难中的生存；语气冷峻平静。")

n = 0
with open(SRC, encoding='utf-8') as f, open(OUT, 'w', encoding='utf-8') as g:
    for line in f:
        item = json.loads(line)
        t = item['text']
        if len(t) < 80:
            continue
        head = t[:80]
        rec = {'conversations': [
            {'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': f'请用余华的风格接着写：\n{head}'},
            {'role': 'assistant', 'content': t},
        ]}
        g.write(json.dumps(rec, ensure_ascii=False) + '\n')
        n += 1
print(f'written: {OUT} ({n} records)')