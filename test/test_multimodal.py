import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
from floppy import FLOPpyTracker

# Set device to CPU to avoid any cuDNN/Driver issues with tiny shapes
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "google/paligemma-3b-pt-224" # Usiamo il checkpoint reale ma piccolo (o una versione tiny)
# Nota: se hai poca RAM usa "hf-internal-testing/tiny-random-PaliGemmaForConditionalGeneration"

print("\n" + "=" * 60)
print(" FLOPpyTracker: Multimodal Test (PaliGemma VLM)")
print("=" * 60)

# ============================================================
# 1. SETUP MODEL & PROCESSOR
# ============================================================
print(f"\n--- Loading PaliGemma VLM ---")
processor = AutoProcessor.from_pretrained(model_name)
model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float32
).to(device)
model.eval()

# ============================================================
# 2. PREPARE REALISTIC MULTIMODAL INPUT
# ============================================================
# PaliGemma è molto sensibile alle dimensioni: usiamo il suo processor standard
prompt = "caption en"
image = Image.new('RGB', (224, 224), color='blue')

# Il processor si occupa di tutto: padding, ridimensionamento e token speciali
inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

print(f"Input IDs shape  : {inputs['input_ids'].shape} (Text + Image Tokens)")
print(f"Pixel Values shape: {inputs['pixel_values'].shape} (Normalized Image)")

# ============================================================
# 3. RUN FLOPpy TRACKER
# ============================================================
print("\n--- Running Multimodal Forward Pass ---")

tracker = FLOPpyTracker(run_name="vlm_paligemma_test")
tracker.run(model=model)

with torch.no_grad():
    # In PaliGemma la forward è lineare e pulita
    outputs = model(**inputs)

# ============================================================
# 4. REPORT & ANALYSIS
# ============================================================
report = tracker.report()

print("\n" + "=" * 60)
print(" 📊 FLOPpy MULTIMODAL REPORT (PaliGemma)")
print("=" * 60)
print(f"Total Forward FLOPs : {report.model_forward_flop / 1e6:.2f} MFLOPs")
print("-" * 60)

# Calcoliamo i parametri della visione vs linguaggio per capire la distribuzione
vision_params = sum(p.numel() for p in model.vision_tower.parameters())
language_params = sum(p.numel() for p in model.language_model.parameters())

print(f"Vision Tower Params  : {vision_params / 1e6:.2f} M")
print(f"Language Model Params: {language_params / 1e6:.2f} M")
print("-" * 60)
print("🚀 SUCCESS: Multimodal graph executed and tracked successfully!")
