import os, torch
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training
quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained("/root/models/Qwen3.8-27B", trust_remote_code=True, dtype=torch.bfloat16, device_map="auto", quantization_config=quant_cfg)
print("base loaded", flush=True)
model = PeftModel.from_pretrained(model, "/root/poetry-hard/saves/sft-GuCheng-ann", is_trainable=True)
# inspect lora params
lora_trainable = 0
lora_total = 0
for name, p in model.named_parameters():
    if "lora" in name.lower():
        lora_total += p.numel()
        if p.requires_grad:
            lora_trainable += p.numel()
print(f"lora params requires_grad: {lora_trainable} / {lora_total}", flush=True)
# count requires_grad across whole model
tr = sum(1 for n,p in model.named_parameters() if p.requires_grad)
print(f"params with requires_grad=True count: {tr}", flush=True)
t, total = model.get_nb_trainable_parameters()
print(f"TRAINABLE {t/1e6:.2f}M / {total/1e9:.2f}B", flush=True)
print("DONE", flush=True)
