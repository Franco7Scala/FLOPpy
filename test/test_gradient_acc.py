import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from floppy import FLOPpyTracker

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "distilgpt2"
accumulation_steps = 4  # Aggiorniamo i pesi ogni 4 passaggi

print("\n" + "=" * 50)
print(f" FLOPpyTracker: Gradient Accumulation Test ({accumulation_steps} steps)")
print("=" * 50)

# 1. Load Model & Optimizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# 2. Dummy Data (Simuliamo un piccolo batch che si ripete)
inputs = tokenizer("Gradient accumulation is a memory-saving technique.", return_tensors="pt").to(device)
inputs["labels"] = inputs["input_ids"].clone()

# 3. Start Tracker
tracker = FLOPpyTracker(run_name="gradient_accumulation_test")
tracker.run(model=model, optimizer=optimizer)

print(f"\n--- Starting training loop with {accumulation_steps} accumulation steps ---")

optimizer.zero_grad()

for i in range(accumulation_steps):
    # Forward Pass
    outputs = model(**inputs)
    loss = outputs.loss / accumulation_steps  # Normalizziamo la loss

    # Backward Pass (Deve accumulare FLOPs ogni volta)
    loss.backward()

    current_report = tracker.report()
    print(f"Step {i + 1}: Model FLOPs (Forward+Backward) so far: {current_report.model_forward_flop + current_report.model_backward_flop:.2f} MFLOPs")
    print(f"Step {i + 1}: Optimizer FLOPs so far: {current_report.optimizer_flop:.2f} MFLOPs")
    print("-" * 30)

# 4. Optimizer Step (Solo qui i FLOPs dell'optimizer devono schizzare su)
print("--- Triggering Optimizer Step ---")
optimizer.step()

# 5. Final Report
report = tracker.report()

print("\n" + "=" * 50)
print(" 📊 FINAL ACCUMULATION REPORT")
print("=" * 50)
print(f"Total Forward FLOPs  : {report.model_forward_flop / 1e6:.2f} MFLOPs")
print(f"Total Backward FLOPs : {report.model_backward_flop / 1e6:.2f} MFLOPs")
print(f"Total Optimizer FLOPs: {report.optimizer_flop / 1e6:.2f} MFLOPs")
print("-" * 50)
print(f"🔥 OVERALL TOTAL     : {report.overall_flop / 1e6:.2f} MFLOPs")
