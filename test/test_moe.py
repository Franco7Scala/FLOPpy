import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from floppy import FLOPpyTracker

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# We use a tiny Mixtral model. It has the exact MoE architecture (8 experts, top-2 routing)
# but scaled down drastically so it runs instantly on any machine.
model_name = "hf-internal-testing/tiny-random-MixtralForCausalLM"

print("\n" + "=" * 50)
print(" FLOPpyTracker: Mixture of Experts (MoE) Routing Test")
print("=" * 50)

# ============================================================
# 1. SETUP MODEL & TOKENIZER
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"\n--- Loading MoE Model ({model_name}) ---")
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()

# Let's count the TOTAL parameters loaded in memory
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Model Parameters: {total_params:,}")

# ============================================================
# 2. RUN FLOPpy TRACKER
# ============================================================
tracker = FLOPpyTracker(run_name="moe_routing_test")
tracker.run(model=model)

prompt = "Mixture of Experts architecture routes tokens to"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
num_tokens = inputs["input_ids"].numel()

print(f"\n--- Running Forward Pass (Tokens: {num_tokens}) ---")

# Forward pass (this triggers the router and conditionally executes experts)
with torch.no_grad():
    outputs = model(**inputs)

# ============================================================
# 3. ANALYSIS & REPORT
# ============================================================
report = tracker.report()

print("\n" + "=" * 50)
print(" 📊 FLOPpy MoE DYNAMIC EXECUTION REPORT")
print("=" * 50)
print(f"Tracked Forward FLOPs : {report.model_forward_flop / 1e6:.2f} MFLOPs")

# Calculate the "Naive" Dense FLOPs.
# If this was a standard dense Transformer, the rule of thumb is:
# Dense FLOPs ≈ 2 * Total Parameters * Number of Tokens
naive_dense_flops = 2 * total_params * num_tokens
print(f"Naive Dense FLOPs     : {naive_dense_flops / 1e6:.2f} MFLOPs")

print("-" * 50)
# The magic of MoE: Did FLOPpy realize we only used 2 out of 8 experts?
if report.model_forward_flop < naive_dense_flops:
    savings = (1 - (report.model_forward_flop / naive_dense_flops)) * 100
    print(f"🚀 SUCCESS: FLOPpy accurately tracked the MoE routing!")
    print(f"   The model used {savings:.1f}% FEWER FLOPs than its parameter count suggests.")
else:
    print("❌ WARNING: FLOPpy counted all experts as if it were a dense model.")
