import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from floppy import FLOPpyTracker

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "distilgpt2"

print("\n" + "=" * 50)
print(" FLOPpyTracker: PEFT/LoRA Adapter Test")
print("=" * 50)

# 1. Load the Hugging Face tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# 2. SETUP MODEL & INJECT LORA ADAPTERS
# ============================================================
print("\n--- Loading Base Model and Injecting LoRA ---")
base_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Configuriamo un Adapter LoRA standard
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,               # Rango della matrice (più è basso, meno parametri)
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["c_attn"] # Applichiamo LoRA ai layer di attention di GPT-2
)

# Questa funzione "avvolge" il modello e congela i pesi base
peft_model = get_peft_model(base_model, lora_config)
peft_model.train()

# Stampiamo quanti parametri stiamo effettivamente addestrando
peft_model.print_trainable_parameters()

# Optimizer (lavorerà solo sull'1% dei parametri)
optimizer = torch.optim.AdamW(peft_model.parameters(), lr=1e-4)

# ============================================================
# 3. DUMMY DATA & LOSS
# ============================================================
prompt = "Fine-tuning Large Language Models with LoRA is"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Aggiungiamo le labels uguali agli input_ids per calcolare la CrossEntropy standard di CausalLM
inputs["labels"] = inputs["input_ids"].clone()

# ============================================================
# 4. FLOPpy INTEGRATION & TRAINING STEP
# ============================================================
print("\n--- Running LoRA Step with FLOPpyTracker ---")

# Inizializziamo il tracker sul modello PEFT
tracker = FLOPpyTracker(run_name="lora_finetuning")
tracker.run(model=peft_model, optimizer=optimizer)

optimizer.zero_grad()

# A. Forward Pass & Loss (Hugging Face calcola la loss in automatico se passiamo le labels)
outputs = peft_model(**inputs)
loss = outputs.loss

# B. Backward Pass
loss.backward()

# C. Optimizer Step
optimizer.step()

# ============================================================
# 5. FINAL REPORT
# ============================================================
report = tracker.report()

print("\n" + "=" * 50)
print(" 📊 FLOPpy PEFT/LoRA REPORT")
print("=" * 50)
print(f"Forward FLOPs  : {report.model_forward_flop / 1e6:.2f} MFLOPs")
print(f"Backward FLOPs : {report.model_backward_flop / 1e6:.2f} MFLOPs")
print(f"Optimizer FLOPs: {report.optimizer_flop / 1e6:.2f} MFLOPs")
print("-" * 50)
print(f"🔥 TOTAL COST  : {report.overall_flop / 1e6:.2f} MFLOPs")