import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from floppy import FLOPpyTracker

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "distilgpt2"

print("\n" + "=" * 60)
print(" FLOPpyTracker: KV Cache (Prefill vs Decode) Inference Test")
print("=" * 60)

# ============================================================
# 1. SETUP MODEL & TOKENIZER
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()

# We use a moderately long prompt to make the Prefill cost obvious
prompt = (
    "In the context of Large Language Models, the KV Cache is an optimization "
    "technique that stores previously computed Keys and Values to avoid "
    "redundant matrix multiplications during autoregressive generation. "
    "This means that"
)
inputs = tokenizer(prompt, return_tensors="pt").to(device)
prompt_length = inputs["input_ids"].shape[1]

print(f"\n--- Loading Model and Prompt (Length: {prompt_length} tokens) ---")

# ============================================================
# 2. RUN FLOPpy TRACKER
# ============================================================
tracker = FLOPpyTracker(run_name="kv_cache_test")
tracker.run(model=model)

# ============================================================
# PHASE 1: PREFILL (Processing the entire prompt)
# ============================================================
print("\n[ PHASE 1: PREFILL ]")
with torch.no_grad():
    # use_cache=True tells the model to return the KV Cache
    outputs = model(**inputs, use_cache=True)

# Get the FLOPs just for this massive parallel step
report_prefill = tracker.report()
prefill_flops = report_prefill.model_forward_flop

print(f"Cost to process {prompt_length} tokens all at once : {prefill_flops / 1e6:.2f} MFLOPs")

# Prepare for Decoding: extract the KV Cache and the next token to feed
past_key_values = outputs.past_key_values
next_token_logits = outputs.logits[:, -1, :]
next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

# ============================================================
# PHASE 2: DECODING (Token-by-token generation using Cache)
# ============================================================
print("\n[ PHASE 2: DECODING (Autoregressive) ]")
num_decode_steps = 5
decode_flops_list = []
previous_total_flops = prefill_flops

for step in range(num_decode_steps):
    with torch.no_grad():
        # Notice we only pass ONE token (next_token) and the Cache!
        outputs = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True
        )

    # Calculate incremental FLOPs (Current Total - Previous Total)
    current_total_flops = tracker.report().model_forward_flop
    step_flops = current_total_flops - previous_total_flops
    decode_flops_list.append(step_flops)
    previous_total_flops = current_total_flops

    # Generate the actual word for logging
    generated_word = tokenizer.decode(next_token[0])
    print(f"Step {step + 1} | Generated: '{generated_word:<10}' | Cost: {step_flops / 1e6:.2f} MFLOPs")

    # Update state for the next step
    past_key_values = outputs.past_key_values
    next_token_logits = outputs.logits[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

# ============================================================
# 3. FINAL COMPARISON
# ============================================================
avg_decode_flops = sum(decode_flops_list) / len(decode_flops_list)

print("\n" + "=" * 60)
print(" 📊 INFERENCE EFFICIENCY REPORT (TTFT vs TPOT)")
print("=" * 60)
print(f"Prefill Cost (First Token)   : {prefill_flops / 1e6:.2f} MFLOPs")
print(f"Average Decode Cost (1 Token): {avg_decode_flops / 1e6:.2f} MFLOPs")
print("-" * 60)

if prefill_flops > avg_decode_flops:
    ratio = prefill_flops / avg_decode_flops
    print(f"🚀 SUCCESS: Decoding a token is {ratio:.1f}x CHEAPER than the Prefill!")
    print(f"   FLOPpy successfully tracked the computational savings of the KV Cache.")
