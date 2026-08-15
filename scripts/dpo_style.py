# -*- coding: utf-8 -*-
"""Qwen3.8-27B 三诗人风格 DPO 后训练（基于 SFT adapter）。

用法:
  python dpo_style.py --style GuCheng --train data/poetry_dpo_qwen38.jsonl \
      --base /root/models/Qwen3.8-27B --adapter /root/poetry-hard/saves/sft-GuCheng \
      --out /root/poetry-hard/saves/dpo-GuCheng

DPO 偏好：chosen=真实诗人诗，rejected=AI 生成诗（Qwen3.8-27B 生成的 500 条 hard 样本构造）。
"""
import argparse
import json
import os
import torch

def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", required=True, help="GuCheng / Haizi / LiBai")
    ap.add_argument("--train", required=True, help="DPO 偏好数据 jsonl")
    ap.add_argument("--base", default="/root/models/Qwen3.8-27B")
    ap.add_argument("--adapter", required=True, help="SFT LoRA adapter 目录")
    ap.add_argument("--out", default="/root/poetry-hard/saves/dpo")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=768)
    args = ap.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig, TrainingArguments, TrainerCallback
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

    rows = load_rows(args.train)
    print(f"[{args.style}] DPO rows: {len(rows)}", flush=True)

    processor = AutoProcessor.from_pretrained(args.base, trust_remote_code=True)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    # Policy: base + SFT LoRA
    model = AutoModelForCausalLM.from_pretrained(
        args.base, trust_remote_code=True,
        dtype=torch.bfloat16, device_map="auto",
        quantization_config=quant_cfg,
    )
    if os.path.isdir(args.adapter):
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # Reference: base + SFT LoRA（冻结），加载到 CPU/低显存
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.base, trust_remote_code=True,
        dtype=torch.bfloat16, device_map="cpu",
        quantization_config=quant_cfg,
        low_cpu_mem_usage=True,
    )
    if os.path.isdir(args.adapter):
        ref_model = PeftModel.from_pretrained(ref_model, args.adapter, is_trainable=False)
    ref_model.eval()

    # 构造 DPO dataset
    from datasets import Dataset

    def build_pair(rec):
        human = next((m["value"] for m in rec["conversations"] if m.get("from") == "human"), "")
        chosen = rec["chosen"]["value"] if isinstance(rec.get("chosen"), dict) else rec["chosen"]
        rejected = rec["rejected"]["value"] if isinstance(rec.get("rejected"), dict) else rec["rejected"]
        return human, chosen, rejected

    def tokenize_pair(batch):
        results = {"prompt_input_ids": [], "prompt_attention_mask": [],
                   "chosen_input_ids": [], "chosen_attention_mask": [],
                   "rejected_input_ids": [], "rejected_attention_mask": []}
        for i in range(len(batch["conversations"])):
            human, chosen, rejected = build_pair({
                "conversations": batch["conversations"][i],
                "chosen": batch["chosen"][i],
                "rejected": batch["rejected"][i],
            })
            # prompt 模板（不含 assistant 部分）
            msgs_prompt = [{"role": "user", "content": human}]
            prompt_text = processor.apply_chat_template(
                msgs_prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            enc_p = processor(prompt_text, return_tensors="pt", truncation=True, max_length=args.max_len)
            enc_c = processor(chosen, return_tensors="pt", truncation=True, max_length=args.max_len)
            enc_r = processor(rejected, return_tensors="pt", truncation=True, max_length=args.max_len)
            results["prompt_input_ids"].append(enc_p["input_ids"][0])
            results["prompt_attention_mask"].append(enc_p["attention_mask"][0])
            results["chosen_input_ids"].append(enc_c["input_ids"][0])
            results["chosen_attention_mask"].append(enc_c["attention_mask"][0])
            results["rejected_input_ids"].append(enc_r["input_ids"][0])
            results["rejected_attention_mask"].append(enc_r["attention_mask"][0])
        return results

    data = {
        "conversations": [r["conversations"] for r in rows],
        "chosen": [r["chosen"] for r in rows],
        "rejected": [r["rejected"] for r in rows],
    }
    ds = Dataset.from_dict(data)
    ds = ds.map(tokenize_pair, batched=True, remove_columns=["conversations", "chosen", "rejected"])
    ds.set_format("torch")

    out_dir = f"{args.out}-{args.style}"
    os.makedirs(out_dir, exist_ok=True)

    class EpochSummaryCallback(TrainerCallback):
        def __init__(self, total_epochs):
            self.total_epochs = total_epochs
            self.last_epoch = 0

        def on_log(self, args, state, control, logs=None, **kwargs):
            epoch = logs.get("epoch")
            if epoch is not None and int(epoch) > self.last_epoch:
                self.last_epoch = int(epoch)
                loss = logs.get("loss")
                if loss is not None:
                    print(f"epoch={self.last_epoch}/{self.total_epochs} train_loss={loss:.6f}", flush=True)

    train_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        save_steps=100,
        save_total_limit=2,
        save_only_model=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    # 使用 TRL DPOTrainer
    from trl import DPOTrainer
    dpo_args = {
        "beta": args.beta,
        "max_length": args.max_len,
        "max_prompt_length": 256,
        "disable_dropout": True,
    }
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=train_args,
        train_dataset=ds,
        tokenizer=tokenizer,
        callbacks=[EpochSummaryCallback(args.epochs)],
        **dpo_args,
    )
    dpo_trainer.train()
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[{args.style}] DPO DONE -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
