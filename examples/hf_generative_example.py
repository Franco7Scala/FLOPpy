import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from floppy import FLOPpyTracker

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "distilgpt2"
prompt = "The future of artificial intelligence"

base_tokenizer = AutoTokenizer.from_pretrained(model_name)

if base_tokenizer.pad_token is None:
    base_tokenizer.pad_token = base_tokenizer.eos_token

# ============================================================
# MODE 1
# ============================================================

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.config.pad_token_id = base_tokenizer.pad_token_id

tracker = FLOPpyTracker(run_name="hf_generate_test")
tracker.run(model=model, tokenizer=base_tokenizer)

inputs = tracker.tokenizer(
    prompt,
    return_tensors="pt",
)
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    generated = model.generate(
        **inputs,
        max_new_tokens=20,
        pad_token_id=base_tokenizer.pad_token_id,
    )

report = tracker.report()

# ============================================================
# MODE 2: context-manager style
# ============================================================

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.config.pad_token_id = base_tokenizer.pad_token_id

with FLOPpyTracker(run_name="hf_generate_test_with") as tracker:
    tracker.start(model=model, tokenizer=base_tokenizer)

    inputs = tracker.tokenizer(
        prompt,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=20,
            pad_token_id=base_tokenizer.pad_token_id,
        )

    tracker.stop()

report = tracker.report()
