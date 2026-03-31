import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from floppy import FLOPpyTracker


# Set device (BitsAndBytes 4-bit requires CUDA)
device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "distilgpt2"
prompt = "The future of green artificial intelligence is FLOPpy!"

# Load the Hugging Face tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# 1. PROFILING STANDARD MODEL (FP32/FP16)
# ============================================================
print("\n--- Running Standard Model (Full Precision) ---")

model_std = AutoModelForCausalLM.from_pretrained(model_name).to(device)

tracker_std = FLOPpyTracker(run_name="standard_model")
tracker_std.run(model=model_std, tokenizer=tokenizer)

# Tokenize using the tracker-wrapped tokenizer
inputs_std = tracker_std.tokenizer(prompt, return_tensors="pt").to(device)

# OPTION 2: Single forward pass (prefill only) to guarantee identical FLOPs
with torch.no_grad():
    model_std(**inputs_std)

report_std = tracker_std.report()
print(report_std)

# Clean up memory for the next model
del model_std
torch.cuda.empty_cache()

# ============================================================
# 2. PROFILING QUANTIZED MODEL (4-BIT)
# ============================================================
print("\n--- Running Quantized Model (4-bit via bitsandbytes) ---")

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

model_quant = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

tracker_quant = FLOPpyTracker(run_name="quantized_model")
tracker_quant.run(model=model_quant, tokenizer=tokenizer)

# Tokenize using the tracker-wrapped tokenizer
inputs_quant = tracker_quant.tokenizer(prompt, return_tensors="pt").to(device)

# OPTION 2: Single forward pass (prefill only) to guarantee identical FLOPs
with torch.no_grad():
    model_quant(**inputs_quant)

report_quant = tracker_quant.report()
print(report_quant)

# ============================================================
# FINAL COMPARISON
# ============================================================
print("\n" + "=" * 30)
print("VERIFICATION SUMMARY")
print("=" * 30)
print(f"Standard  - FLOPs: {report_std.overall_flop:,} | BOPs: {report_std.overall_bop:,}")
print(f"Quantized - FLOPs: {report_quant.overall_flop:,} | BOPs: {report_quant.overall_bop:,}")
print("-" * 30)

if report_std.overall_flop == report_quant.overall_flop:
    print("SUCCESS: FLOPs remain identical (same algorithmic complexity).")
else:
    print("WARNING: FLOPs differ (check if input lengths are identical).")

if report_quant.overall_bop < report_std.overall_bop:
    reduction = (1 - (report_quant.overall_bop / report_std.overall_bop)) * 100
    print(f"SUCCESS: BOPs reduced by {reduction:.2f}% (lower hardware effort).")
