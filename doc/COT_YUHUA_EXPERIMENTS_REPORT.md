# MiniMind CoT 优化与余华风格模型实验报告

> 日期：2026-08-18
> 范围：① 顾城 AR 模型思维链(CoT)优化与后训练；② 余华 13 册全集解析与余华风格模型训练
> 结论先行：CoT 数据构建与训练管线全部打通，AR-CoT 模型学会了「读题→理解→成诗」的三段式结构（100/100 合格）；余华模型学会 CoT 前缀结构（【构思】【基调】【正文】）但正文长文连贯性受 104M 容量限制，属诚实记录的中等质量结果。

---

## 1. 顾城 AR-CoT 优化（任务 2）

### 1.1 CoT 模板与数据

模板（三段式思维链）：

```
【读题】《题目》让我想到<意象>构成的画面，<情感>的气息藏在字句之间……
【意象】意象词列表
【情感基调】<情感词>
【诗】
<诗歌正文>
```

- 数据：`dataset/sft_gucheng_cot.jsonl`（607 条）
  - 213 条来自既有 SFT 样本（assistant 回答前加 CoT 前缀）
  - 394 条来自 pretrain 真诗扩充（带严格诗歌过滤：无英文/乱码/散文长句/元数据）
  - 全部 CoT 前缀由规则从**真实诗文本特征**生成（意象词表匹配、情感词表匹配），不虚构
- 构建脚本：`scripts/build_gucheng_cot.py`

### 1.2 训练

```
python train_full_sft.py --save_weight full_sft_gucheng_cot --epochs 10 \
  --batch_size 8 --learning_rate 1e-5 --data_path ../dataset/sft_gucheng_cot.jsonl \
  --from_weight full_sft_gucheng --max_seq_len 512 --save_interval 100 --num_workers 0
```

- 基础权重：`full_sft_gucheng`（未加 CoT 的最佳 AR 模型）
- 10 epochs，loss 3.02 → 2.04
- 产物：`out/full_sft_gucheng_cot_768.pth`（137MB）
- **工程教训**：Windows 下 DataLoader `num_workers=8` 会导致训练卡死（GPU 利用率 ~1%、无日志、权重不更新），改 `--num_workers 0` 后立即恢复 98% GPU 利用率。此前多次"训练卡死"实为此问题。

### 1.3 生成评估（100 样本）

```
python scripts/gen_gucheng_batch.py --model ar --n 100 --version cot_full --weight full_sft_gucheng_cot
```

- 结果：100/100 合格（共生成 112 条，过滤 12 条，合格率同未加 CoT 版本）
- 模型正确学会 CoT 三段式前缀输出
- **发现的问题：读题行题目错位**（如 prompt《一代人》→ 读题写成《你走了的书》）
  - 根因：104M 模型在采样温度 ~1.0 下，对"读题行必须原样引用题目"的复制能力不足，读题行书名号内容从题目分布漂移
  - 修复（解码约束，无需重训）：`scripts/gen_gucheng_batch.py` 新增 `fix_title_drift()`，生成后强制将【读题】行第一个书名号替换为题目原文
- 诗歌正文质量：具备顾城式简洁意象，但个别句子存在语法不通（104M 容量限制）

### 1.4 其他后训练尝试（评估）

| 尝试 | 状态 | 结论 |
|---|---|---|
| CoT SFT（full SFT，10 epochs） | ✅ 完成 | 学会 CoT 结构；读题错位用解码约束修复 |
| 指令增强 SFT v2（6 epochs） | ✅ 完成（阴性结果） | 系统提示加入"读题必须以题目原文开头"强化指令，从基础权重只训 6 epochs → CoT 结构退化（出现 `</think>`、数字、无书名号引用），30 样本测试中 27 条无法匹配读题模式。结论：104M 小模型对复杂指令鲁棒性差，训练 epoch 数不足时结构先崩；**解码约束（fix_title_drift）是更可靠的修复** |
| DPO | 暂缓 | 需高质量偏好对（chosen/rejected），当前无可靠裁判信号，噪音大 |
| LoRA | 暂缓 | 与 full SFT 收益重叠，且 104M 小模型 full 微调成本低 |

---

## 2. 余华全集解析与风格模型（任务 3）

### 2.1 epub 解析

- 输入：`余华作品全集（全13册）.epub`（2.3MB，297 条目/286 html）
- 脚本：`scripts/parse_yuhua_epub.py`（spine 顺序 + ncx 树遍历定位 13 册边界 + html 逐 `<p>`/`<h>` 提取）
- 输出：`dataset/yuhua_raw.jsonl`（20952 段），字段 `{book, book_index, chapter, chapter_index, segment_index, text}`
  - segment_index 全书连续（QA 修复过每章重置的问题）；书籍边界预计算 `book_spine_start`（QA 修复过跳目录索引错位）

各册段数：

| 册 | 段数 |
|---|---|
| 活着 | 2038 |
| 许三观卖血记 | 2465 |
| 兄弟 | 5547 |
| 在细雨中呼喊 | 2103 |
| 黄昏里的男孩 | 1639 |
| 战栗 | 1385 |
| 没有一条道路是重复的 | 539 |
| 世事如烟 | 808 |
| 温暖和百感交集的旅程 | 461 |
| 我胆小如鼠 | 1586 |
| 鲜血梅花 | 829 |
| 现实一种 | 1166 |
| 音乐影响了我的写作 | 386 |

### 2.2 余华风格 CoT 设计

余华风格要素：白描冷峻、短句、重复、荒诞与苦难并存、儿童/弱者视角、让事实自己说话。

模板：

```
【构思】这一段承接<书名>的叙述。写底层人物在苦难中的生存，语气克制，不煽情。……
【基调】冷峻克制，叙事不动声色。
【正文】
<余华原文段落>
```

- 数据：`dataset/sft_yuhua_cot.jsonl`（522 条）+ `dataset/pretrain_yuhua.jsonl`（18793 条）
- 构建脚本：`scripts/build_yuhua_dataset.py`；CoT 前缀由真实特征规则生成，正文全部为余华原文

### 2.3 训练

阶段一 Pretrain（warm-start 自 pretrain_gucheng）：

```
python train_pretrain.py --save_weight pretrain_yuhua --epochs 2 --batch_size 8 \
  --learning_rate 1e-4 --data_path ../dataset/pretrain_yuhua.jsonl \
  --from_weight pretrain_gucheng --max_seq_len 512 --num_workers 0
```

阶段二 SFT（CoT）：

```
python train_full_sft.py --save_weight full_sft_yuhua_cot --epochs 10 --batch_size 8 \
  --learning_rate 1e-5 --data_path ../dataset/sft_yuhua_cot.jsonl \
  --from_weight pretrain_yuhua --max_seq_len 512 --num_workers 0
```

- 产物：`out/pretrain_yuhua_768.pth`、`out/full_sft_yuhua_cot_768.pth`

### 2.4 生成评估（100 样本）

```
python scripts/gen_yuhua_batch.py --weight full_sft_yuhua_cot --n 100
```

- 结果：100/100 通过格式过滤（长 40-700、无英文/数字/乱码/重复）
- **学会的部分**：CoT 结构（【构思】【基调】【正文】）稳定输出；【构思】句式为真实余华特征规则的变体
- **不足（诚实记录）**：
  - 正文长文连贯性差，常出现荒诞拼贴（如混入《兄弟》人物李光头/林红到其他主题）
  - 个别样本正文过早截断、CoT 前缀后无完整正文
  - 根因：104M 模型 + 522 条 CoT 数据的长文本叙事容量瓶颈；短诗（顾城）与长篇散文（余华）的生成难度量级不同
- 改进方向：更大的基座模型（7B LoRA）、更多 CoT 数据（从 yuhua_raw 扩充 2000+ 条）、更长的 pretrain

---

## 3. 磁盘与工程约束落实

- C 盘已全部迁出（`gucheng_upload`→E 盘 `minimind/upload_archive/`，Git 仓库→E 盘），后续一律不写 C 盘
- 所有训练权重保存在 E 盘 `minimind/out/`，当前 E 盘剩余 ~30GB，权重单文件 ~137MB
- Windows 训练坑：DataLoader `num_workers>0` 卡死 → 一律 `--num_workers 0`；`max_seq_len` 768 风险 → 统一 512

---

## 4. 产物清单

| 产物 | 路径 |
|---|---|
| 顾城 CoT 数据 | `dataset/sft_gucheng_cot.jsonl`（607 条） |
| 顾城 AR-CoT 权重 | `out/full_sft_gucheng_cot_768.pth` |
| 顾城 CoT 生成样本 | `out/gucheng_samples_ar_cot_full.jsonl`（100 条） |
| 余华原始分段 | `dataset/yuhua_raw.jsonl`（20952 段） |
| 余华 pretrain 数据 | `dataset/pretrain_yuhua.jsonl`（18793 条） |
| 余华 CoT 数据 | `dataset/sft_yuhua_cot.jsonl`（522 条） |
| 余华模型权重 | `out/pretrain_yuhua_768.pth`、`out/full_sft_yuhua_cot_768.pth` |
| 余华生成样本 | `out/yuhua_samples_cot_v1.jsonl`（100 条） |
