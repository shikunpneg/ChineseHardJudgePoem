# ChineseHardJudgePoem 项目技术文档（v2.0 第二版）

> 目标：构建中文诗歌风格模仿 + 难例判别（Hard Judge）的完整流水线。包含 4 个诗人风格微调模型、中英混合负样本约束 SFT、以及模仿-原作配对的难例数据集 v2（4450 条）。

---

## 1. 项目概述

本项目用 **Qwen2.5-3B-Instruct** 作为基座，在 4 位中文诗人（顾城、海子、海子中文强化、李白）的诗作上做 **QLoRA 4bit 有监督微调**，获得风格化生成能力。随后用微调后的模型"指定一首原作→模仿创作一首新诗"，构造 [模仿诗, 被模仿原作] 1:1 配对的难例数据集。该数据集用于评测判别模型"区分背诵/抄袭 vs 模仿/同风格新作"的能力，也可作为人标平台的无答案输入。

仓库：`https://github.com/shikunpneg/ChineseHardJudgePoem`（本地：`e:\生成诗歌\ChineseHardJudgePoem\`）

---

## 2. 模型体系

### 2.1 基座模型

| 项目 | 值 |
|------|-----|
| 基座 | `Qwen/Qwen2.5-3B-Instruct`（中英文 30 亿参数指令微调版）|
| 架构 | Decoder-only Transformer，35 层，32 头，hidden=2048 |
| 上下文窗口 | 32K |
| 原始权重路径 | `e:\生成诗歌\poetry-gen-train\models\Qwen2.5-3B-Instruct\` |

> 选择 3B 的原因：在 RTX 4060 Laptop 8GB 显存条件下，QLoRA 4bit 量化后仅占用 ~2.2GB 显存，batch_size=1 可训练；fp16 推理仅 ~5.5GB。同时 3B 指令模型对中文 prompt 理解和风格模仿的质量明显优于更小的 1.5B。

### 2.2 四个诗人风格微调模型

| 模型路径 | 目标诗人 | 体裁 | 训练数据量 | 关键特性 |
|----------|----------|------|-----------|----------|
| `models/Qwen2.5-3B-GuCheng` | 顾城 | 现代诗（朦胧派） | `gucheng_train.jsonl` | 短句、意象密集、童话气质 |
| `models/Qwen2.5-3B-Haizi` | 海子 | 现代诗（麦地/太阳） | `haizi_train.jsonl` | 麦地/村庄/远方/姐姐意象 |
| `models/Qwen2.5-3B-Haizi-CN` | 海子(中文约束) | 现代诗 | `haizi_train.jsonl` + 中英混合负样本 SFT | **强制全中文**，解决 v1 英文混入问题 |
| `models/Qwen2.5-3B-LiBai` | 李白 | 古体诗（五/七言、古风、乐府） | `libai_train.jsonl` 838 首 | 乐府歌行、古风、绝句律诗 |

> **Haizi-CN 的特殊性**：在 Haizi 基础上追加一轮"中英混合负样本 + 纯中文正样本"的有监督对碰训练，大幅降低输出中的英文泄漏（Haizi 组 v1 英文混入约 48%，Haizi-CN v2 为 0）。

### 2.3 HuggingFace 已发布模型

- `shikunpunk/Qwen2.5-3B-GuCheng`
- `shikunpunk/Qwen2.5-3B-Haizi`
- `shikunpunk/Qwen2.5-3B-Haizi-CN`
- `shikunpunk/Qwen2.5-3B-LiBai`

全部采用 4bit 量化导出 + safetensors 分片，加载时配合 bitsandbytes NF4。

---

## 3. 训练方式（QLoRA 4bit）

所有风格微调均通过 **LLaMA-Factory** 的 `llamafactory-cli train` 命令执行，配置 YAML 位于 `llama_data/train_<model>.yaml`。

### 3.1 量化与分布式策略

| 项目 | 值 |
|------|-----|
| 量化方法 | QLoRA 4bit（bitsandbytes `NF4` dtype）|
| 计算 dtype | bfloat16（匹配 CUDA 11.8 + torch 2.5.1）|
| LoRA Target | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`（全部线性层）|
| LoRA Rank (r) | 32 |
| LoRA Alpha | 64（= r × 2）|
| LoRA Dropout | 0.05 |

### 3.2 优化器与学习率

| 项目 | GuCheng / Haizi / Haizi-CN | LiBai |
|------|---------------------------|-------|
| 优化器 | `paged_adamw_8bit` | `paged_adamw_8bit` |
| 初始学习率 | `1e-4` | `1e-4` |
| 调度器 | cosine with warmup | cosine with warmup |
| warmup 步数 | 总步数的 10% | 总步数的 10% |
| 训练轮数 (epoch) | **3** | **10**（李白数据量小，多轮收敛）|
| batch_size (per device) | 1 | 1 |
| gradient_accumulation_steps | **16** | **8** |
| gradient_checkpointing | on | on |
| max_grad_norm | 1.0 | 1.0 |
| NEFTune Noise | 5 | 5 |

### 3.3 数据格式（ShareGPT）

```json
{
  "conversations": [
    {"from": "system", "value": "你是诗人海子。只会输出中文诗歌..."},
    {"from": "human",  "value": "请写一首关于春天的诗。"},
    {"from": "gpt",    "value": "面朝大海，春暖花开\n..."}
  ],
  "meta": {
    "author": "haizi",
    "title": "无题",
    "genre": "现代诗"
  }
}
```

**硬约束**：
- 所有样本标题必须统一为 **"无题"**（避免标题→作者特征泄漏）。
- System prompt 内嵌诗人身份和体裁约束。
- Haizi-CN 额外追加"**只使用中文，禁止出现任何英文字母或单词**"。

### 3.4 中英混合负样本 SFT（Haizi-CN 核心）

针对 v1 中 Haizi 生成出现 48% 英文混入的问题，构造 **1:1 的纯中文正样本 + 中英混合负样本** 对碰 SFT：

**负样本示例（判断）：**
```json
{
  "conversations": [
    {"from": "system", "value": "你是诗歌判别器。请判断下面文本是否符合中文诗歌的纯中文要求。"},
    {"from": "human",  "value": "请判断以下文本是否属于诗歌：\n我有一间house，面朝sea..."}
    {"from": "gpt",    "value": "不是诗歌。"}
  ]
}
```

**正样本示例（判断）：**
```json
{
  "conversations": [
    {"from": "system", "value": "你是诗歌判别器。..."},
    {"from": "human",  "value": "请判断以下文本是否属于诗歌：\n从明天起，做一个幸福的人..."}
    {"from": "gpt",    "value": "是诗歌。"}
  ]
}
```

> 关键：正负样本的 **human 指令完全中性**（"请判断以下文本是否属于诗歌"），不使用"判断是否有英文"等直接暗示标签的语句——防止模型学格式捷径而非真正的语言判别能力。最后在生成时，此判别能力通过 system prompt 转移到 Haizi-CN 自约束输出上。

### 3.5 导出流程

训练完成后用 `llamafactory-cli export export_<model>.yaml`：
1. LoRA adapter + 基座权重合并
2. 量化为 **4bit safetensors**（`load_in_4bit=true, bnb_4bit_quant_type=nf4`）
3. 写入 `models/Qwen2.5-3B-<Author>/`
4. `huggingface_hub.upload_folder` 推送到对应 HF 仓库
5. 验证远程文件齐全后删除本地临时缓存 `.cache/huggingface/upload/`

---

## 4. v2 难例数据集构建流水线

### 4.1 v1 存在的问题（v2 修复点）

| 问题 | v1 情况 | v2 修复 |
|------|---------|---------|
| 配对错乱 | 从共享标题池随机抽标题，ref_text 不是真正的被模仿诗 | prompt 内嵌被模仿诗全文，确保生成 = 对这首诗的模仿 |
| 英文混入 | GuCheng 35%、Haizi 48%、LiBai 35% | Haizi-CN 用中英对碰 SFT + 所有模型生成时 `EN_RE = [A-Za-z]` 过滤 + 最多 8 轮重试 |
| 体裁误标 | 四组全部标为"现代诗" | 按模型本身标记：顾城/海子→现代诗，李白→古体诗 |
| LiBai 池污染 | 混入书信、赋、碑、颂、记、序、并序等非诗歌 | 黑名单 30 条 + 标题后缀黑名单 `{赋,碑,记,诔,祭,铭,墓志}` + "并序"剔除 + "兮">=2 判定为骚体赋排除 + 评注段截断（`strip_libai_commentary()` 遇"按""此诗""注曰"截断）|
| 非诗歌对 | v1 混入"模仿诗 vs 非诗歌文本"的错误对 | v2 **100% 保证**：每组样本 = [模型模仿诗, 被模仿的诗人原作] 1:1 配对 |
| 无重试 | 截断/英文样本直接写入，污染数据 | BATCH=12 批量生成 + 校验失败样本进 `pending_fail` 下一轮重试，最多 8 轮 |
| 无断点续跑 | 中断从头来 | 基于 `done_count = Counter(prompt)` 按出现次数跳过已完成条数，支持池复用场景精确续跑 |

### 4.2 步骤一：构建被模仿诗作池（`build_pools_v2.py`）

输入文件来自 `llama_data/{gucheng,haizi,libai,haizi_train}_{train,test,style}.jsonl`，合并 train+test+style 获得全量诗作。

**清洗流水线：**

```
原始诗作（ShareGPT meta 取 ref_text + 标题）
  → 去重：(title, text[:80]) 双 key 去重
  → 纯中文：剔除含 [A-Za-z] 的作品（李白全集脚注/英译残留）
  → 诗歌性（现代诗 + 古体诗共用门槛）：
      · 3 <= 句均字数 <= 14
      · 超长句（>32 字）占比 <= 15%
      · 总行数 >= 3
  → 李白额外规则：
      · 黑名单 30 条（《为赵宣城与杨右相书》等书信）
      · 标题后缀黑名单：赋/碑/记/诔/祭/铭/墓志
      · "X书"保留例外：书怀/书情/书筒/书字/书斋/秘书（其余"X书"判定为书信）
      · 标题中含"并序"直接剔除
      · "兮"字 >= 2 判定为骚体赋，剔除
      · 评注段截断：遇评注关键词"按""此诗""注曰""诗曰""解题""写作背景"即刻截断
  → 标题存在性校验：必须有《XX》标题
```

**池规模（v2）：**
| 诗人 | 池大小（首）| 说明 |
|------|-----------|------|
| 顾城 GuCheng | 180 | 合并 gucheng_train+test+style |
| 海子 Haizi | 124 | 合并 haizi_train+test+style |
| 海子 Haizi-CN | 124 | 与 Haizi 共用池 |
| 李白 LiBai | 630 | 从 825 首经评注截断 + 严格过滤后保留 |

### 4.3 步骤二：生成模仿任务输入（`build_hard_inputs_v2.py`）

每组固定 **1250 条输入**（李白当前版本只跑到 700 条，见 §4.6）。采用均匀循环采样 + `SEED=20260813`：

```python
rng = np.random.default_rng(SEED)
for _ in range(GROUP_SIZE):
    poem = rng.choice(pool)  # 随机抽取一首被模仿诗
    prompt = build_prompt(author, poem["title"], poem["text"], genre)
```

**Prompt 模板（李白古体诗版）：**
```
你是诗人李白。下面是李白创作的古体诗《{title}》：
{ref_text}
请模仿这首诗的风格与格律，创作一首全新的古体诗（题目自拟）。要求：
(1)全诗只用中文，不要出现任何英文；
(2)只输出诗歌正文，不要输出题目以外的任何解释、题跋、作者署名、注释或引用；
(3)不要出现'按''此诗''注曰''诗曰'等评注开头。
```

输出：`llama_data/hard_input_v2.jsonl`（5000 条设计规模），每条字段：
```
{model, title, genre, prompt, ref_text}
```

### 4.4 步骤三：批量生成 + 校验 + 重试（`gen_hard_v2.py`）

**生成参数：**
```
BATCH = 12
MAX_NEW_TOKENS = 284（现代诗 220 / 古体诗放宽到 284）
MAX_RETRY = 8
temperature = 1.0
top_p = 0.9
repetition_penalty = 1.05
pad_token_id = tokenizer.eos_token_id
do_sample = True
```

**`is_valid_poem()` 校验（不合格进重试）：**
```python
def is_valid_poem(text, genre="现代诗"):
    if EN_RE.search(text):              return False, "english"
    if len(text) < 12:                  return False, "too_short"
    if genre == "古体诗" and len(text) > 800: return False, "too_long"
    if genre == "现代诗" and len(text) > 600: return False, "too_long"
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if genre == "古体诗" and len(lines) < 2: return False, "too_few_lines"
    if genre == "现代诗" and len(lines) < 3: return False, "too_few_lines"
    if sum(len(l) for l in lines)/len(lines) > 30: return False, "avg_line_too_long"
    # 古体诗评注关键词黑名单
    bad = {"参考答案","诗评","解题","写作背景","译文","原诗如下","作品鉴赏","注：","作者简介"}
    if any(w in text for w in bad):     return False, "commentary_leak"
    return True, "ok"
```

**重试机制（核心算法）：**
```
pending = rows_to_process
fail_tag = Counter()  # 每个 prompt 失败次数
for retry in range(MAX_RETRY):
    batch = pending[:BATCH]; pending = pending[BATCH:]
    generated = model.generate(batch)  # 批量
    for rec, g in zip(batch, generated):
        ok, reason = is_valid_poem(g)
        if ok:
            write_output({...rec, generated: g, status: "ok"})
        else:
            fail_tag[rec["prompt"]] += 1
            if fail_tag[rec["prompt"]] < MAX_RETRY:
                pending.append(rec)      # 下一轮重试
            else:
                write_output({...rec, generated: "（生成失败）", status: "fail"})
```

**断点续跑：**
```python
from collections import Counter
done_count = Counter()
if os.path.exists(out_path):
    for l in open(out_path, encoding="utf-8"):
        rec = json.loads(l)
        done_count[rec["prompt"]] += 1  # 统计该 prompt 已被完成几次
skip = Counter()
keep = []
for r in rows:
    if skip[r["prompt"]] < done_count[r["prompt"]]:
        skip[r["prompt"]] += 1
        continue
    keep.append(r)
rows = keep
```
> 这确保"同一首原作被抽中多次"的场景（池复用）下不会跳过超额或漏跳。

### 4.5 步骤四：合并 + 去重 + 复用统计（`make_hard_dataset_v2.py`）

合并四组文件：
```
hard_gen_v2_GuCheng.jsonl  (1250)
hard_gen_v2_Haizi.jsonl    (1250)
hard_gen_v2_Haizi-CN.jsonl (1285 → 取 1250，去重)
hard_gen_v2_LiBai.jsonl    (700  → 过滤失败 → 约 630)
```

**处理：**
1. 过滤 `generated == "（生成失败）"` 的样本
2. Haizi-CN 1285 条 → 按 `(prompt, generated)` 去重 → 取前 1250 条
3. 按 `(prompt, generated)` 全局去重（四组间不会有重复，因为 prompt 带诗人身份）
4. 计算相似度：
   - `sim_jaccard`：字符 2-gram Jaccard 相似度
   - `sim_cosine`：TfidfVectorizer（char 2-4 gram）余弦相似度
5. 统计复用：`unique_prompts_count`, `avg_reuse_times`
6. 输出：
   - `hard_dataset_v2_XXXX.jsonl`（最终总集）
   - `hard_dataset_v2_report.md`（统计表 + 说明）

### 4.6 v2 实际规模

LiBai 组提前终止在 700 条（含约 70 个失败，成功约 630 条），不再跑完 1250。最终总集规模：

| 组 | 成功条数 | 说明 |
|----|---------|------|
| GuCheng | 1250 | 1250 全成功 |
| Haizi | 1250 | 1250 全成功 |
| Haizi-CN | 1250 | 从 1285 去重后取 1250 |
| LiBai | 630 | 700 过滤失败后 ≈ 630 |
| **合计** | **≈ 4380** | 标注为 v2.0 **4450 版**（实际精确数由合并脚本输出）|

> 如果后续需要补满 5000，只需把 LiBai 输入中未生成的条目接着用断点续跑跑完 1250，再跑合并脚本即可。

---

## 5. 质量评估指标

### 5.1 生成时自动校验（见 §4.4）
- english=0：无任何英文字母
- fail=0：合并后无"生成失败"标记
- 形式校验：字数范围、行数范围、平均行长度范围

### 5.2 相似度指标
每条样本计算模仿诗 ↔ 原作的两个相似度：

| 指标 | 方法 | 高相似度含义 |
|------|------|-------------|
| `sim_jaccard` | 字符 2-gram 的 Jaccard 交集/并集 | 字面重合度高（可能背诵/照抄）|
| `sim_cosine` | TfidfVectorizer（字符 2-4 gram）+ 余弦 | 词频/字面模式接近 |

**v2 目标相似度区间（非背诵非完全无关）：**
- `0.01 < sim_jaccard < 0.3`：典型模仿创作区间
- `sim_jaccard > 0.6`：高度疑似背诵/复制，送入人工标注

### 5.3 人标流程
- 在最终 4450 条中过滤背诵样本（`sim_jaccard > 0.6` 或 `sim_cosine > 0.6`）
- 剩余高相似度 + 随机抽取低相似度的样本构成 `to_annotate.jsonl`
- 上传至标注网站：每条样本呈现 [模仿诗, 原作]，标注员判断"背诵/抄袭/改写/全新创作"
- 标注无"标准答案"（无标签噪声）

---

## 6. 环境与依赖

| 项目 | 版本/要求 |
|------|----------|
| 操作系统 | Windows 10/11（bitsandbytes 需 `pytorch_env` conda 环境的 0.48.x）|
| Python | 3.11 (`D:\anacoda\envs\pytorch_env\`)，运行参数 `-X utf8` |
| PyTorch | 2.5.1 + CUDA 11.8 |
| transformers | 4.57.3（**注意 5.8.x 有 tokenizer 严重 bug**，monkey-patch 见下）|
| bitsandbytes | 0.48.x（兼容 torch 2.5.1 + CUDA 11.8）|
| LLaMA-Factory | 适配 Qwen2.5 的版本 |
| accelerate | 最新版 |
| peft | 最新版 |
| huggingface_hub | 最新版 |
| GPU | NVIDIA RTX 4060 Laptop 8GB GDDR6（最低 7GB 空闲磁盘空间用于 4bit 导出）|
| 磁盘 | `C:/temp_models/` 临时导出（E 盘曾空间不足）|
| 命令 cwd | 所有 `RunCommand` 必须显式 `-cwd "e:\生成诗歌\..."` 防止沙箱走 F 盘 |

### 6.1 已知 Bug 与 Workarounds

| Bug | 现象 | Workaround |
|-----|------|-----------|
| **transformers 5.8.x tokenizer 崩溃** | `extra_special_tokens` 中混入非 dict 元素，tokenizer 初始化崩溃 | **保留在 4.57.3 不升级**；如强制升级，需 monkey-patch `PreTrainedTokenizerBase._add_tokens()` 跳过非 dict 元素 |
| **bitsandbytes 0.48.x 安装失败** | `pytorch_env` 以外的 conda 环境装上也没用 | `pip install bitsandbytes==0.48.x` 只在 `pytorch_env` 内执行 |
| **4bit 导出磁盘不足** | 中间 checkpoint 撑爆 E 盘 | 导出到 `C:/temp_models/`，上传 HF 后删除 |
| **推理 CPU/CUDA 张量不匹配** | `input_ids` 在 CPU，model 在 CUDA | `input_ids = input_ids.to(model.device)` |
| **3D tensor 报错** | `apply_chat_template` 已返回 batch 维度，再 `.unsqueeze(0)` 变成 3D | `apply_chat_template` 后**不要再 unsqueeze** |
| **HF upload cache 膨胀** | `.cache/huggingface/upload/` 几十 GB 残留 | 每次上传后手动删除 |
| **overwrite_output_dir 报错** | LLaMA-Factory 检测到残留 checkpoint | train YAML 必须设置 `overwrite_output_dir: true` |

---

## 7. 代码/目录速览

```
e:\生成诗歌\poetry-gen-train\
├── models\                              # 导出的 4bit 权重（不在 git）
│   ├── Qwen2.5-3B-Instruct\             # 基座
│   ├── Qwen2.5-3B-GuCheng\
│   ├── Qwen2.5-3B-Haizi\
│   ├── Qwen2.5-3B-Haizi-CN\
│   └── Qwen2.5-3B-LiBai\
├── llama_data\                          # ShareGPT 训练集/配置
│   ├── dataset_info.json                # 数据集注册
│   ├── hard_input_v2.jsonl              # 5000 条模仿任务输入
│   ├── gucheng_train.jsonl / test.jsonl # 各诗人 SFT 数据
│   ├── haizi_train.jsonl / test.jsonl
│   ├── libai_train.jsonl / test.jsonl
│   ├── train_<m>.yaml / export_<m>.yaml # LLaMA-Factory 配置
│   └── v2_pool_GuCheng.jsonl ...        # 各诗人 v2 被模仿池
├── scripts\
│   ├── build_pools_v2.py                # 步骤一：构建被模仿池
│   ├── build_hard_inputs_v2.py          # 步骤二：生成 5000 任务输入
│   ├── gen_hard_v2.py                   # 步骤三：批量生成+校验+重试+断点
│   ├── make_hard_dataset_v2.py          # 步骤四：合并去重+统计+报告
│   ├── build_negative_data.py           # 中英混合负样本构造
│   ├── prepare_haizi_cn_sft.py          # Haizi-CN SFT 准备（中英约束）
│   ├── prepare_libai_sft.py             # LiBai SFT 准备
│   ├── upload_*.py                      # HF 上传脚本
│   ├── eval_*.py / experiment_*.py      # 评测/实验
│   └── make_paper_figures.py / figures/ # 论文图表
├── hard_data\                           # v2 中间产物（不在 git）
│   ├── hard_gen_v2_GuCheng.jsonl (1250)
│   ├── hard_gen_v2_Haizi.jsonl   (1250)
│   ├── hard_gen_v2_Haizi-CN.jsonl (1285)
│   ├── hard_gen_v2_LiBai.jsonl    (700)
│   ├── hard_dataset_v2_4450.jsonl       # 最终总集
│   ├── hard_dataset_v2_report.md        # 统计报告
│   └── hard_stats_v2_*.json             # 各模型 jaccard/cosine/en 统计
├── paper\                               # 研究论文（markdown + PDF）
│   ├── research_paper.md
│   └── research_paper.pdf               # xelatex + pandoc 生成
└── saves\                               # LoRA 训练 checkpoint（不在 git）

e:\生成诗歌\ChineseHardJudgePoem\         # ← GitHub 上传目标
├── README.md                            # 项目说明 + v2 数据集创建流程
├── TECHNICAL_DOC_v2.md                  # ← 本文档
├── data\                                # v2 最终数据
├── paper\                               # 研究论文 PDF
├── scripts\                             # 关键脚本镜像
└── hard_data\                           # v2 报告 + 统计
```

---

## 8. 已知限制与后续改进

| 限制 | 说明 | 改进方向 |
|------|------|---------|
| **池复用** | 顾城全集仅 180 首、海子 124 首、李白 630 首，每组 1250 条必然循环复用（平均复用 6-10×）| 扩展诗人全集中文 OCR；引入多位同风格诗人合成风格池 |
| **LiBai 组 28 首诗始终失败** | 即使 8 轮重试，《湖边采莲妇》等 28 首李白诗作始终生成不出合格古体诗（疑似模型对某些句式没学会）| 从池中移除这 28 首；或增加针对性 SFT 补例 |
| **仅字符级相似度** | sim_jaccard/sim_cosine 均基于字符 n-gram，无法识别语义等价但字不同的改写 | 接入 `BERT-CCPoem`（THUNLP 开源古诗句向量 BERT，92.6 万诗预训练）做语义相似度 |
| **基座 3B 有限** | 复杂长诗（李白《忆旧游寄谯郡元参军》等）模仿能力有限 | 测试 7B/9B 基座如 `ricardozhy/Qwen1.5-7B-poem`、`wnwu/Qwen3.5-9B-gelv-poet` 对比 |
| **无平仄/格律硬校验** | 李白绝句律诗可能出现平仄不协 | 接入格律校验（如 `gelv` 库）加入 is_valid_poem |

---

## 9. 版本标记

- **v1.0**：5000 条，含配对错乱、英文混入、体裁误标、LiBai 非诗歌池等问题（已弃用，仅作对比基线）
- **v2.0 第二版**：≈ 4380-4450 条，100% 模仿-原作 1:1 配对，纯中文，形式校验通过，LiBai=700 条（本文档版本）
