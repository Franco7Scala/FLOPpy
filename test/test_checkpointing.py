import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from floppy import FLOPpyTracker

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "distilgpt2"

print("\n" + "=" * 60)
print(" FLOPpyTracker: Gradient Checkpointing (Memory/Compute Tradeoff)")
print("=" * 60)

# ============================================================
# 1. SETUP MODEL & DATA
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model (use_cache=False is REQUIRED for gradient checkpointing)
model = AutoModelForCausalLM.from_pretrained(model_name, use_cache=False).to(device)
model.train()

prompt = "Gradient checkpointing saves memory by recomputing activations during the backward pass."
inputs = tokenizer(prompt, return_tensors="pt").to(device)
inputs["labels"] = inputs["input_ids"].clone()

# ============================================================
# 2. STANDARD TRAINING (NO CHECKPOINTING)
# ============================================================
print("\n--- Running Standard Training (No Checkpointing) ---")
optimizer_std = torch.optim.AdamW(model.parameters(), lr=1e-4)
tracker_std = FLOPpyTracker(run_name="standard_training")

tracker_std.run(model=model, optimizer=optimizer_std)

optimizer_std.zero_grad()
outputs_std = model(**inputs)
loss_std = outputs_std.loss
loss_std.backward()
optimizer_std.step()

report_std = tracker_std.report()

# ============================================================
# 3. CHECKPOINTED TRAINING (RECOMPUTATION)
# ============================================================
print("\n--- Running Training WITH Gradient Checkpointing ---")
# Reset gradients and clear cache
optimizer_std.zero_grad()
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ENABLE CHECKPOINTING
model.gradient_checkpointing_enable()

# We need a new optimizer instance and a new tracker
optimizer_chk = torch.optim.AdamW(model.parameters(), lr=1e-4)
tracker_chk = FLOPpyTracker(run_name="checkpointed_training")

tracker_chk.run(model=model, optimizer=optimizer_chk)

optimizer_chk.zero_grad()
outputs_chk = model(**inputs)
loss_chk = outputs_chk.loss
loss_chk.backward()
optimizer_chk.step()

report_chk = tracker_chk.report()

# ============================================================
# 4. FINAL COMPARISON
# ============================================================
flops_std = report_std.overall_flop
flops_chk = report_chk.overall_flop

print("\n" + "=" * 60)
print(" 📊 GRADIENT CHECKPOINTING EFFICIENCY REPORT")
print("=" * 60)
print(f"Standard Total FLOPs     : {flops_std / 1e6:.2f} MFLOPs")
print(f"Checkpointed Total FLOPs : {flops_chk / 1e6:.2f} MFLOPs")
print("-" * 60)

if flops_chk > flops_std:
    increase = ((flops_chk / flops_std) - 1) * 100
    print(f"🚀 SUCCESS: FLOPpy caught the recomputation ghost-work!")
    print(f"   The checkpointed run executed {increase:.1f}% MORE FLOPs.")
    print(f"   (Theoretical expectation is ~33% increase)")
else:
    print("❌ WARNING: FLOPpy did not detect the extra forward passes inside the backward pass.")
    print("   Check if the backward hook is intercepting dynamically generated nodes.")
