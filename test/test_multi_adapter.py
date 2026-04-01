import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from floppy import FLOPpyTracker

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "distilgpt2"

print("\n" + "=" * 50)
print(" FLOPpyTracker: Multi-Adapter (LoRA Duo) Test")
print("=" * 50)

# 1. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# 2. SETUP MODEL & INJECT MULTIPLE ADAPTERS
# ============================================================
print("\n--- Loading Base Model and Injecting Multiple Adapters ---")
base_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Configurazione Adapter 1 (R=8)
lora_config_1 = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    target_modules=["c_attn"]
)
model = get_peft_model(base_model, lora_config_1, adapter_name="adapter_skills_A")

# Aggiungiamo un secondo Adapter (R=16)
lora_config_2 = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    target_modules=["c_attn"]
)
model.add_adapter("adapter_skills_B", lora_config_2)

# ATTENZIONE: set_adapter attiva un adapter per il forward pass.
# Ma per il nostro test di FLOPpy, vogliamo che entrambi siano "trainable".
model.set_adapter("adapter_skills_B")
model.train()

# Per simulare un vero scenario multi-adapter dove entrambi sono pronti all'uso,
# ci assicuriamo che i parametri di "adapter_skills_A" non siano stati congelati
# dal passaggio a "adapter_skills_B".
for name, param in model.named_parameters():
    if "lora_" in name:
        param.requires_grad = True

# Ora stampiamo i parametri: vedrai che il totale dei trainable è la somma di A + B
model.print_trainable_parameters()

# Optimizer (gestirà i pesi di ENTRAMBI gli adapter)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# ============================================================
# 3. DATA & TRAINING STEP
# ============================================================
prompt = "Multi-adapter training allows for"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
inputs["labels"] = inputs["input_ids"].clone()

print("\n--- Running Multi-Adapter Step with FLOPpyTracker ---")

tracker = FLOPpyTracker(run_name="multi_adapter_test")
tracker.run(model=model, optimizer=optimizer)

optimizer.zero_grad()

# Forward Pass (Entrambi gli adapter contribuiscono al calcolo)
outputs = model(**inputs)
loss = outputs.loss

# Backward Pass
loss.backward()

# Optimizer Step
optimizer.step()

# ============================================================
# 4. FINAL REPORT
# ============================================================
report = tracker.report()

print("\n" + "=" * 50)
print(" 📊 FLOPpy MULTI-ADAPTER REPORT")
print("=" * 50)
print(f"Forward FLOPs  : {report.model_forward_flop / 1e6:.2f} MFLOPs")
print(f"Backward FLOPs : {report.model_backward_flop / 1e6:.2f} MFLOPs")
print(f"Optimizer FLOPs: {report.optimizer_flop / 1e6:.2f} MFLOPs")
print("-" * 50)
print(f"🔥 TOTAL COST  : {report.overall_flop / 1e6:.2f} MFLOPs")
