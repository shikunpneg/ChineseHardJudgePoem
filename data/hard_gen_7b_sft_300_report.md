# Qwen1.5-7B-Poem-SFT 300 条合格仿写样本

## 1. 概况

- 总样本数：**300**（LiBai 75 + Haizi 75 + Haizi-CN 75 + GuCheng 75，各作者完全均衡）
- 生成模型：`shikunpunk/Qwen1.5-7B-Poem-SFT`（Qwen1.5-7B 领域基座 + LoRA adapter，checkpoint-75，SFT eval_loss=2.941）
- 每样本 = **[仿写诗作(generated), 被模仿诗作(ref_text)] 1:1 配对**，严格同诗人
- 参照池：v2_pool（与 hard_dataset_v2 相同），排除 SFT 训练集 598 条与基座对比 24 条标题
- 生成配置：`temperature=1.0, top_p=0.9, repetition_penalty=1.05, max_new_tokens=512`
- 内嵌 QA：生成后即时校验（≥30 字 / 纯中文 / 零回显），不合格自动重抽重试

## 2. 为什么引入 7B

此前 4 个 3B 模型（Qwen2.5-3B）在 8GB 笔记本显卡上可训练，但诗艺质量有限。为提升仿写质量，
对 `ricardozhy/Qwen1.5-7B-poem` 领域基座做 QLoRA 4-bit SFT（598 条仿写样本），得到本 7B-SFT 模型。
相比 3B：意象更丰富、长诗完整度更高；相比未微调的 7B 基座（24/24 含英文、19/24 带 JSON 尾巴、3/24 主题跑偏），
SFT 后英文混入 / JSON 尾巴 / 主题跑偏均为 0。

## 3. 质量校验

| 检查项 | 结果 |
|--------|------|
| 总样本数 | 300（4 作者各 75） |
| 含英文字母 | 0 |
| 含 JSON 尾巴 | 0 |
| 过短（<30 字） | 0 |
| 回显痕迹（Human/Assistant/``` 等） | 0 |
| 重复标题（同作者内） | 0 |
| 与 SFT 训练集文本重叠 | 0（参考诗与生成文本均无重叠） |

字数：min=31，median=114，max=906；行数：median=11。

## 4. 数据格式

```json
{
  "model": "Qwen1.5-7B-Poem-SFT",
  "author": "LiBai",
  "title": "静夜思",
  "genre": "古体诗",
  "prompt": "请仔细阅读下面这首《静夜思》（作者：LiBai）……",
  "ref_text": "……被模仿真实诗作……",
  "generated": "……仿写诗作全文……",
  "temperature": 1.0,
  "elapsed_sec": 96.3,
  "chars": 118,
  "lines": 14
}
```

## 5. 说明

- 生成脚本：`poetry-gen-train/scripts/gen_300_sft.py`（支持断点续跑，跳过已完成样本）
- 截断规则：`poetry-gen-train/scripts/trim_sft_outputs.py`（检测英文/JSON/对话痕迹，截断在命中行之前）
- 推理方式：4-bit base + PeftModel 直接加载，无需合并模型
- 本批 300 条为**合格样本**，可直接用于诗歌真伪判别 / 风格保真度评估的测试与扩充
