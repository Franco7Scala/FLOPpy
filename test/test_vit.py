import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from peft import LoraConfig, get_peft_model, TaskType
from floppy import FLOPpyTracker

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "google/vit-base-patch16-224"

print("\n" + "=" * 50)
print(" FLOPpyTracker: ViT + LoRA Adapter Test")
print("=" * 50)

# 1. Setup Image Processor (equivalente del tokenizer per le immagini)
# In un test dummy, ci serve principalmente per conoscere le dimensioni di input
processor = ViTImageProcessor.from_pretrained(model_name)

# ============================================================
# 2. SETUP ViT MODEL & INJECT LORA ADAPTERS
# ============================================================
print("\n--- Loading ViT Base Model and Injecting LoRA ---")
base_model = ViTForImageClassification.from_pretrained(model_name).to(device)

# Rimuoviamo TaskType o lo lasciamo vuoto
lora_config = LoraConfig(
    # task_type non è obbligatorio per i modelli di visione in molte versioni di PEFT
    r=16,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "value"]
)

# Applichiamo LoRA
peft_model = get_peft_model(base_model, lora_config)
peft_model.train()

# Stampiamo i parametri addestrabili (molto pochi rispetto al totale)
peft_model.print_trainable_parameters()

# Optimizer
optimizer = torch.optim.AdamW(peft_model.parameters(), lr=1e-4)

# ============================================================
# 3. DUMMY IMAGE DATA & LOSS
# ============================================================
# Creiamo un'immagine dummy: Batch=1, Canali=3, H=224, W=224
pixel_values = torch.randn(1, 3, 224, 224).to(device)
labels = torch.tensor([1]).to(device) # Una classe a caso per calcolare la loss

inputs = {
    "pixel_values": pixel_values,
    "labels": labels
}

# ============================================================
# 4. FLOPpy INTEGRATION & TRAINING STEP
# ============================================================
print("\n--- Running ViT LoRA Step with FLOPpyTracker ---")

# Inizializziamo il tracker
tracker = FLOPpyTracker(run_name="vit_lora_test")
tracker.run(model=peft_model, optimizer=optimizer)

optimizer.zero_grad()

# A. Forward Pass (ViT calcola la loss internamente se passiamo 'labels')
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
print(" 📊 FLOPpy ViT-LoRA REPORT")
print("=" * 50)
print(f"Forward FLOPs  : {report.model_forward_flop / 1e9:.2f} GFLOPs")
print(f"Backward FLOPs : {report.model_backward_flop / 1e9:.2f} GFLOPs")
print(f"Optimizer FLOPs: {report.optimizer_flop / 1e6:.2f} MFLOPs")
print("-" * 50)
print(f"🔥 TOTAL COST  : {report.overall_flop / 1e9:.2f} GFLOPs")
