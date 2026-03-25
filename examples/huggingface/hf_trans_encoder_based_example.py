import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from floppy import FLOPpyTracker


# Set device for computation
device = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================
# MODEL AND TOKENIZER SETUP
# ==========================================

model_name = "distilbert-base-uncased"
texts = [
    "FLOP estimation is important for green AI.",
    "Tracking compute helps improve model efficiency.",
]

# Load the Hugging Face tokenizer
base_tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the pre-trained Sequence Classification model and move it to the device
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)


# ==========================================
# FLOPpy INTEGRATION (MODE 1)
# ==========================================

# 1. Initialize the tracker with a custom experiment name
tracker = FLOPpyTracker(run_name="hf_trans_encoder_test")

# 2. Start monitoring.
# By passing the tokenizer alongside the model, FLOPpy will also track
# the computational workload required for text preprocessing (tokenization).
tracker.run(model=model, tokenizer=base_tokenizer)


# ==========================================
# TEXT PROCESSING AND INFERENCE
# ==========================================

# Encode the input texts.
# CRITICAL: Use `tracker.tokenizer` instead of the base tokenizer!
# This allows FLOPpy to intercept and count the tokenization operations (preproc_ops).
# Padding and truncation are automatically handled.
inputs = tracker.tokenizer(
    texts,
    padding=True,
    truncation=True,
    return_tensors="pt",
)

# Move the tokenized inputs (input_ids, attention_mask) to the correct hardware device
inputs = {k: v.to(device) for k, v in inputs.items()}

# Perform a standard forward pass for sequence classification.
# FLOPpy dynamically intercepts the tensor dimensions (batch size, sequence length)
# to compute the exact FLOPs/BOPs of the Transformer's attention mechanisms and linear layers.
with torch.no_grad():
    outputs = model(**inputs)


# ==========================================
# FINAL REPORT
# ==========================================

# 3. Generate and print the detailed hardware and algorithmic workload summary
report = tracker.report()
print(report)
