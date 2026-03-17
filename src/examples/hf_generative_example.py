import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from floppy import FLOPpyTracker

# ------------------------------------------------------------
# Shared setup
# ------------------------------------------------------------

model_name = "distilgpt2"
prompt = "The future of artificial intelligence"

base_tokenizer = AutoTokenizer.from_pretrained(model_name)

# ============================================================
# MODE 1
# ============================================================

model = AutoModelForCausalLM.from_pretrained(model_name)

tracker = FLOPpyTracker(
    run_name="hf_generate_test",
    print_summary=True,
    print_hardware=True,
)

tracker.run(model=model, tokenizer=base_tokenizer)

inputs = tracker.tokenizer(
    prompt,
    return_tensors="pt",
)

with torch.no_grad():
    generated = model.generate(**inputs, max_new_tokens=20)

print(tracker.report())

# ============================================================
# MODE 2: context manager style
# ============================================================

model = AutoModelForCausalLM.from_pretrained(model_name)

with FLOPpyTracker(
    run_name="hf_generate_test_with",
    print_summary=True,
    print_hardware=True,
) as tracker:

    tracker.start(model=model, tokenizer=base_tokenizer)

    inputs = tracker.tokenizer(
        prompt,
        return_tensors="pt",
    )

    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=20)

    tracker.stop()

print(tracker.report())
