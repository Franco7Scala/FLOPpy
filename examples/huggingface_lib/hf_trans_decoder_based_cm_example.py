import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from floppy import FLOPpyTracker


# Set device for computation
device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# MODEL AND TOKENIZER SETUP
# ==========================================

model_name = "distilgpt2"
prompt = "The future of green artificial intelligence is FLOPpy!"

# Load the Hugging Face tokenizer
base_tokenizer = AutoTokenizer.from_pretrained(model_name)

# Ensure the tokenizer has a padding token defined (required by some HF models)
if base_tokenizer.pad_token is None:
    base_tokenizer.pad_token = base_tokenizer.eos_token

# Load the pre-trained Causal Language Model and move it to the device
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.config.pad_token_id = base_tokenizer.pad_token_id

# ==========================================
# FLOPpy INTEGRATION (CONTEXT MANAGER)
# ==========================================

# Using FLOPpyTracker as a context manager (with statement) ensures safe
# execution and clean scope management for your profiling session.
with FLOPpyTracker(run_name="hf_trans_decoder_test") as tracker:
    # 1. Start monitoring.
    # By passing the tokenizer alongside the model, FLOPpy will also track
    # the computational workload required for text preprocessing.
    tracker.start(model=model, tokenizer=base_tokenizer)

    # ==========================================
    # TEXT GENERATION (INFERENCE)
    # ==========================================

    # Encode the input prompt.
    # CRITICAL: Use `tracker.tokenizer` instead of the base tokenizer!
    # This allows FLOPpy to intercept and count the tokenization operations.
    inputs = tracker.tokenizer(prompt, return_tensors="pt")

    # Move the tokenized inputs to the correct hardware device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Perform text generation.
    # FLOPpy will automatically track the FLOPs/BOPs across all the internal
    # autoregressive forward passes executed under the hood by `generate`.
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=20,
            pad_token_id=base_tokenizer.pad_token_id,
        )

    # Explicitly stop the tracking before exiting the context manager scope.
    tracker.stop()

# ==========================================
# FINAL REPORT
# ==========================================

# 2. Generate and print the detailed hardware and algorithmic workload summary.
# This is safely called outside the 'with' block.
report = tracker.report()
print(report)
