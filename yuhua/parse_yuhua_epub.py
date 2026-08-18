# -*- coding: utf-8 -*-
"""解析余华 13 册全集 epub → 按书籍/章节/段落顺序输出 yuhua_raw.jsonl
字段: {book, book_index, chapter, chapter_index, segment_index, text}
"""
import zipfile, re, json, sys, html as html_mod
import xml.etree.ElementTree as ET

EPUB = r"E:\生成诗歌\余华作品全集（全13册） (余华) (z-library.sk, 1lib.sk, z-lib.sk).epub"
OUT = r"E:\生成诗歌\dataset\yuhua_raw.jsonl"
NCX_NS = {"x": "http://www.daisy.org/z3986/2005/ncx/"}

with zipfile.ZipFile(EPUB) as z:
    names = z.namelist()
    opf = z.read("EPUB/content.opf").decode("utf-8", "ignore")
    ncx = z.read("EPUB/toc.ncx").decode("utf-8", "ignore")

# ---------- 1. spine 顺序 ----------
spine_ids = re.findall(r'<itemref[^>]*idref="([^"]+)"', opf)
man = {}
for mm in re.finditer(r'<item[^>]*id="([^"]+)"[^>]*href="([^"]+)"', opf):
    man[mm.group(1)] = mm.group(2)
for mm in re.finditer(r'<item[^>]*href="([^"]+)"[^>]*id="([^"]+)"', opf):
    man[mm.group(2)] = mm.group(1)
spine_files = []
for sid in spine_ids:
    h = man.get(sid)
    if h:
        spine_files.append(h.replace("EPUB/", "").split("#")[0])
print("spine files:", len(spine_files))

# ---------- 2. 解析 ncx 书/章节层级 ----------
root = ET.fromstring(ncx)
navs = root.find("x:navMap", NCX_NS)

def walk(nav, depth, out):
    label = nav.find("x:navLabel/x:text", NCX_NS)
    src = nav.find("x:content", NCX_NS)
    label_t = "".join(label.itertext()).strip() if label is not None else ""
    src_t = src.get("src", "") if src is not None else ""
    out.append((depth, label_t, src_t))
    for child in nav.findall("x:navPoint", NCX_NS):
        walk(child, depth + 1, out)

flat = []
for nav in navs.findall("x:navPoint", NCX_NS):
    walk(nav, 0, flat)

# 分割成 book 列表：depth==0 为书，其下 depth>0 为章节
books = []
i = 0
while i < len(flat):
    if flat[i][0] == 0:
        bname = flat[i][1]
        bsrc = flat[i][2].split("#")[0]
        j = i + 1
        chapters = []
        while j < len(flat) and flat[j][0] > 0:
            chapters.append((flat[j][1], flat[j][2]))
            j += 1
        books.append({"name": bname, "src": bsrc, "chapters": chapters})
        i = j
    else:
        i += 1
print("books:", len(books))
for b in books:
    print("  ", b["name"], "|", b["src"], "| chapters:", len(b["chapters"]))

# ---------- 3. 章节标题表: file -> [(title, anchor)] ----------
chapter_map = {}  # file -> list of (title, src_full)
for b in books:
    for ctitle, csrc in b["chapters"]:
        f = csrc.split("#")[0]
        chapter_map.setdefault(f, []).append(ctitle)

# ---------- 4. 逐文件解析 ----------
def clean_text(s):
    s = s.replace("\r", "").replace("\u200b", "").replace("\ufeff", "")
    s = html_mod.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def parse_html(t):
    """返回 [(tag, text), ...] 提取 h1-h6 和 p"""
    items = []
    # 用正则按块提取,保留顺序
    for mm in re.finditer(r"<h[1-6][^>]*>(.*?)</h[1-6]>|<p[^>]*>(.*?)</p>", t, re.S | re.I):
        if mm.group(1) is not None:
            items.append(("h", clean_text(mm.group(1))))
        else:
            items.append(("p", clean_text(mm.group(2))))
    return items

recs = []
book_order = [b["name"] for b in books if b["name"] != "目录"]
# 预计算每个 book 的 spine 起始位置（按 spine 文件顺序）
book_spine_start = []
for b in books:
    if b["name"] == "目录":
        book_spine_start.append(None)
        continue
    if b["src"] in spine_files:
        book_spine_start.append(spine_files.index(b["src"]))
    else:
        # 找不到起始文件：用第一章 src
        f = b["chapters"][0][1].split("#")[0]
        book_spine_start.append(spine_files.index(f) if f in spine_files else None)

for bi, b in enumerate(books):
    if b["name"] == "目录":
        continue
    start_idx = book_spine_start[bi]
    if start_idx is None:
        print("WARN: book src not in spine:", b["name"])
        continue
    # 本书文件区间 = [start_idx, 下一本书 start_idx)
    end_idx = len(spine_files)
    for bj in range(bi + 1, len(books)):
        if book_spine_start[bj] is not None:
            end_idx = book_spine_start[bj]
            break
    own_files = spine_files[start_idx:end_idx]
    chapter_idx = 0
    seg_idx = 0
    cur_chapter = ""
    for f in own_files:
        with zipfile.ZipFile(EPUB) as z:
            try:
                raw = z.read("EPUB/" + f).decode("utf-8", "ignore")
            except KeyError:
                continue
        items = parse_html(raw)
        if not items:
            continue
        for tag, text in items:
            if not text:
                continue
            if tag == "h":
                # 章节标题（优先 ncx 名称匹配）
                ncx_titles = chapter_map.get(f, [])
                if ncx_titles:
                    # 匹配该标题在 ncx 中的顺序
                    k = 0
                    for k, t0 in enumerate(ncx_titles):
                        if t0 == text:
                            break
                    cur_chapter = ncx_titles[min(k, len(ncx_titles) - 1)]
                else:
                    cur_chapter = text
                chapter_idx += 1
                continue
            # p 段落
            recs.append({
                "book": b["name"],
                "book_index": book_order.index(b["name"]) + 1,
                "chapter": cur_chapter or (b["chapters"][0][0] if b["chapters"] else ""),
                "chapter_index": chapter_idx,
                "segment_index": seg_idx,
                "text": text,
            })
            seg_idx += 1

print("total segments:", len(recs))
# 按书籍统计
from collections import Counter
cnt = Counter(r["book"] for r in recs)
for k, v in cnt.items():
    print(f"  {k}: {v}")

# ---------- 5. 写出 ----------
import os
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("written:", OUT, "bytes:", os.path.getsize(OUT))
