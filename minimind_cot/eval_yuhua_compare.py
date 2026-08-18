# -*- coding: utf-8 -*-
"""余华场景三模型对比评测：统计可读性、重复率、结构完整性等指标
用法: python scripts/eval_yuhua_compare.py
"""
import json, re, os, math

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FILES = {
    'AR-CoT': f'{ROOT}/out/yuhua_cmp_ar_v1.jsonl',
    'dLM':    f'{ROOT}/out/yuhua_cmp_dllm_v1.jsonl',
    'Linear': f'{ROOT}/out/yuhua_cmp_linear_v1.jsonl',
}
# 生成统计（从日志记录：总生成数/过滤数）
GEN_STATS = {'AR-CoT': (30, 0), 'dLM': (72, 42), 'Linear': (31, 1)}


def repetition_score(t):
    """基于 6-gram 检测的整体重复率（短句循环也能捕获）"""
    flat = re.sub(r'\s', '', t)
    n = 6
    if len(flat) < n:
        return 0.0
    grams = [flat[i:i + n] for i in range(len(flat) - n + 1)]
    uniq = len(set(grams))
    return 1.0 - uniq / len(grams)


def content_repetition(t):
    """正文部分重复率（去掉【构思】【基调】结构行）"""
    if '【正文】' in t:
        t = t.split('【正文】', 1)[1]
    return repetition_score(t)


def longest_clean_run(t):
    """最长无重复片段（近似可读性：越长的无循环重复片段越可读）"""
    flat = re.sub(r'\s', '', t)
    n = 12
    if len(flat) < n:
        return len(flat)
    grams = [flat[i:i + n] for i in range(len(flat) - n + 1)]
    # 找最长的滑动窗口内无重复 gram 的连续区间
    seen, start, best = set(), 0, 0
    for i, g in enumerate(grams):
        while g in seen:
            seen.discard(grams[start])
            start += 1
        seen.add(g)
        best = max(best, i - start + 1)
    return best + n - 1


def main():
    print(f"{'模型':<10}{'样本':>4}{'通过率':>8}{'均长':>6}{'重复率':>8}{'正文重复':>8}{'最长可读':>8}{'结构完整':>8}")
    for name, path in FILES.items():
        rows = []
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        if not rows:
            continue
        gen, fil = GEN_STATS[name]
        passed = len(rows)
        lengths = [len(r['content']) for r in rows]
        reps = [repetition_score(r['content']) for r in rows]
        creps = [content_repetition(r['content']) for r in rows]
        runs = [longest_clean_run(r['content']) for r in rows]
        complete = sum(1 for r in rows if '【构思】' in r['content'] and '【正文】' in r['content'])
        pass_rate = passed / (gen) * 100
        print(f"{name:<10}{passed:>4}{pass_rate:>7.0f}%{sum(lengths)/len(lengths):>6.0f}"
              f"{sum(reps)/len(reps):>8.2f}{sum(creps)/len(creps):>8.2f}{sum(runs)/len(runs):>8.0f}"
              f"{complete/passed*100:>7.0f}%")


if __name__ == '__main__':
    main()
