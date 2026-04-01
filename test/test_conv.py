import torch
from transformers import AutoImageProcessor, ResNetForImageClassification
from peft import LoraConfig, get_peft_model
from floppy import FLOPpyTracker

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "microsoft/resnet-50"

print("\n" + "=" * 50)
print(" FLOPpyTracker: CNN (ResNet) + LoRA Test")
print("=" * 50)

# 1. Setup Image Processor
processor = AutoImageProcessor.from_pretrained(model_name)

# ============================================================
# 2. SETUP CNN MODEL & INJECT LORA ADAPTERS
# ============================================================
print("\n--- Loading ResNet and Injecting LoRA ---")
base_model = ResNetForImageClassification.from_pretrained(model_name).to(device)

# Configurazione LoRA per CNN
# In ResNet (Hugging Face), i layer target sono tipicamente "convolution"
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    # Applichiamo LoRA a tutti i layer convoluzionali principali
    target_modules=["convolution"],
    modules_to_save=["classifier"], # Spesso si addestra anche l'ultimo layer linear
)

peft_model = get_peft_model(base_model, lora_config)
peft_model.train()

# Verifica dei parametri addestrabili
peft_model.print_trainable_parameters()

# Optimizer
optimizer = torch.optim.AdamW(peft_model.parameters(), lr=1e-4)

# ============================================================
# 3. DUMMY IMAGE DATA
# ============================================================
# Input standard per ResNet: 3x224x224
pixel_values = torch.randn(1, 3, 224, 224).to(device)
labels = torch.tensor([1]).to(device)

inputs = {"pixel_values": pixel_values, "labels": labels}

# ============================================================
# 4. FLOPpy INTEGRATION & TRAINING STEP
# ============================================================
print("\n--- Running CNN LoRA Step with FLOPpyTracker ---")

tracker = FLOPpyTracker(run_name="resnet_lora_test")
tracker.run(model=peft_model, optimizer=optimizer)

optimizer.zero_grad()

# Forward
outputs = peft_model(**inputs)
loss = outputs.loss

# Backward
loss.backward()

# Step
optimizer.step()

# ============================================================
# 5. FINAL REPORT
# ============================================================
report = tracker.report()

print("\n" + "=" * 50)
print(" 📊 FLOPpy CNN-LoRA REPORT")
print("=" * 50)
# ResNet-50 ha circa 4 GigaMACs, quindi ci aspettiamo circa 8-9 GFLOPs nel forward
print(f"Forward FLOPs  : {report.model_forward_flop / 1e9:.2f} GFLOPs")
print(f"Backward FLOPs : {report.model_backward_flop / 1e9:.2f} GFLOPs")
print(f"Optimizer FLOPs: {report.optimizer_flop / 1e6:.2f} MFLOPs")
print("-" * 50)
print(f"🔥 TOTAL COST  : {report.overall_flop / 1e9:.2f} GFLOPs")
