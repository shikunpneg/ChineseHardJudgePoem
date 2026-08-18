# -*- coding: utf-8 -*-
"""构建 sft_yuhua_cot_v2.jsonl（V2 防人物名泄漏版）：
- User prompt 改为通用题面（不出现真实书名/章节/具名人物）
- Assistant 改为：风格定位 + 余华原文段落（不再以"承接《XX》"开头）
- 数据规模 800-1200 条，按 13 册分层采样
- 过滤含余华小说具名人物的段落（许三观/家珍/一乐/福贵/李光头/林红/孙伟/宋刚 等），
  避免 CoT 模板学到这些词的高频分布
"""
import json, os, re, random
from collections import Counter, defaultdict
random.seed(42)

RAW = r"E:\生成诗歌\dataset\yuhua_raw.jsonl"
OUT = r"E:\生成诗歌\minimind\dataset\sft_yuhua_cot_v2.jsonl"

# 余华小说具名人物黑名单（来自 13 册全集常见角色）
NAMED_CHARS = {
    # 活着
    '福贵', '家珍', '凤霞', '有庆', '二喜', '苦根',
    # 许三观卖血记
    '许三观', '许玉兰', '一乐', '二乐', '三乐', '何小勇', '方铁匠',
    # 兄弟
    '李光头', '宋钢', '林红', '宋凡平', '李兰',
    # 在细雨中呼喊
    '孙光平', '孙光林', '孙广才', '鲁鲁',
    # 现实一种
    '皮皮',
    # 我胆小如鼠
    '杨敏', '杨科',
}

STYLE_POOL = [
    "不直接写悲伤，而是写具体的物件和动作，让物件代替情感",
    "用白描手法，冷峻平静地叙述，几乎不带形容词",
    "写底层人物在苦难中的生存，语气克制，不煽情",
    "短句、重复，让事实自己说话，情感藏在细节里",
    "从儿童或弱者视角叙述，冷静中带着荒诞",
]
SCENE_WORDS = ["田野", "街道", "屋子", "河边", "镇上", "城里", "医院", "地里", "厨房", "夜晚", "路上", "屋里", "田间", "村口", "窗口", "树下", "雨中", "雪中", "井边", "桥上", "码头", "河边"]
MOOD_POOL = ["冷峻克制", "平静压抑", "荒诞反讽", "温情隐忍", "沉郁缓慢"]

# 通用题面池（V1 的"续写《活着》某章"是泄漏主因，改成"描写某类场景"）
GENERIC_PROMPTS = [
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
    "请用余华的风格，写一段童年的田野记忆。",
    "请用余华的风格，写一段丧礼上的沉默。",
    "请用余华的风格，写一段饥荒年代的一顿饭。",
    "请用余华的风格，写一段夜色中的河岸。",
    "请用余华的风格，写一段被退回的信。",
    "请用余华的风格，写一段镇上理发店的对话。",
    "请用余华的风格，写一段走夜路的人。",
    "请用余华的风格，写一段母亲的针线活。",
    "请用余华的风格，写一段父子在桥头分手。",
    "请用余华的风格，写一段农忙季节的一天。",
    "请用余华的风格，写一段冬天的早晨。",
    "请用余华的风格，写一段集市上的相遇。",
    "请用余华的风格，写一段老人与孙子的对话。",
    "请用余华的风格，写一段医院走廊的等待。",
    "请用余华的风格，写一段关于孤独的描述。",
    "请用余华的风格，写一段关于命运的思考。",
    "请用余华的风格，写一段关于时间的流逝。",
]

SYSTEM = ("你是一位深谰余华小说风格的写作者。余华的写作特点是：白描、克制、不直接抒情，"
          "用具体的物件和动作呈现情感；写底层人物在苦难中的生存；语气冷峻平静，"
          "偶尔有荒诞与反讽；短句，重复，让事实自己说话。\n\n"
          "要求：只用泛指人物（他/她/我/父亲/母亲/孩子/女人/男人 等），"
          "不出现特定人物的姓名；不引用任何特定作品；保持白描克制。")

def has_named_char(t):
    return any(c in t for c in NAMED_CHARS)

def has_dialogue(t):
    return t.count('“') >= 2 or t.count('"') >= 2

def extract_people(t):
    hits = [w for w in ['他', '她', '我', '爹', '娘', '母亲', '父亲', '爷爷', '奶奶', '儿子', '女儿', '女人', '男人', '孩子'] if w in t[:200]]
    return hits[:3]

def first_scene(t):
    for w in SCENE_WORDS:
        if w in t:
            return w
    return None

def avg_sent_len(t):
    sents = re.split(r'[。！？；\n]', t)
    sents = [s for s in sents if s.strip()]
    if not sents:
        return 0
    return sum(len(s) for s in sents) / len(sents)

# 读取原始段
recs = [json.loads(l) for l in open(RAW, encoding='utf-8')]
print(f'raw segments: {len(recs)}')

# 每本书的段数，按比例采样
sizes = Counter(r['book'] for r in recs)
total = sum(sizes.values())
target_total = 1000  # V2 目标规模
quota = {}
for b, n in sizes.items():
    quota[b] = max(20, min(120, int(target_total * n / total)))

sft = []
book_counts = defaultdict(int)
skip_named = skip_too_short = skip_too_long = skip_quota = 0

for r in recs:
    t = r['text'].strip()
    if len(t) < 80 or len(t) > 400:  # V2 放宽下限要求更长的段落
        skip_too_short += 1
        continue
    if has_named_char(t):  # V2 关键：过滤具名人物段
        skip_named += 1
        continue
    book = r['book']
    if book_counts[book] >= quota[book]:
        skip_quota += 1
        continue
    book_counts[book] += 1

    dial = has_dialogue(t)
    people = extract_people(t)
    scene = first_scene(t)
    avg_len = avg_sent_len(t)
    style = random.choice(STYLE_POOL)
    mood = random.choice(MOOD_POOL)

    # 【构思】改为风格定位，不出现具体书名/人物姓名
    parts = [style + '。']
    if dial:
        parts.append('段中有直接对话，人物开口时语气平淡。')
    else:
        parts.append('段中无人开口，全靠叙述者的眼睛看见什么就写什么。')
    if scene:
        parts.append(f'场景落在{scene}，写的是日常里的一瞬间。')
    if people:
        parts.append(f'叙述围绕{"/".join(people[:2])}展开，写他们的处境，不评判。')
    if avg_len <= 12:
        parts.append('句子短，节奏快，像把话截断了一样。')
    elif avg_len >= 25:
        parts.append('句子长而铺开，但不修饰，仍旧是白描。')
    conceive = ''.join(parts)

    assistant = f'【构思】{conceive}\n【基调】{mood}，叙事不动声色。\n【正文】\n{t}'
    user = random.choice(GENERIC_PROMPTS)
    sft.append({'conversations': [
        {'role': 'system', 'content': SYSTEM},
        {'role': 'user', 'content': user},
        {'role': 'assistant', 'content': assistant},
    ]})

print(f'cot v2 items: {len(sft)}')
print(f'skip: named={skip_named} short/long={skip_too_short} quota={skip_quota}')
print(f'per-book: {dict(book_counts)}')

# 写出
with open(OUT, 'w', encoding='utf-8') as f:
    for item in sft:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
print(f'written: {OUT} ({os.path.getsize(OUT)/1024:.0f}KB)')