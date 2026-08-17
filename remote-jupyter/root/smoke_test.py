import os, torch
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training
print("start load", flush=True)
quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained("/root/models/Qwen3.8-27B", trust_remote_code=True, dtype=torch.bfloat16, device_map="auto", quantization_config=quant_cfg)
print("base loaded", flush=True)
adapter = "/root/poetry-hard/saves/sft-GuCheng-ann"
model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
t, total = model.get_nb_trainable_parameters()
print(f"TRAINABLE {t/1e6:.2f}M / {total/1e9:.2f}B", flush=True)
print("SMOKE_OK", flush=True)
