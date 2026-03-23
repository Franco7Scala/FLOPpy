import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from floppy import FLOPpyTracker

# ------------------------------------------------------------
# Shared setup
# ------------------------------------------------------------

model_name = "distilbert-base-uncased"
texts = [
    "FLOP estimation is important for green AI.",
    "Tracking compute helps improve model efficiency.",
]

base_tokenizer = AutoTokenizer.from_pretrained(model_name)

# ============================================================
# MODE 1
# ============================================================

model = AutoModelForSequenceClassification.from_pretrained(model_name)

tracker = FLOPpyTracker(
    run_name="hf_transformer_test",
)

tracker.run(model=model, tokenizer=base_tokenizer)

inputs = tracker.tokenizer(
    texts,
    padding=True,
    truncation=True,
    return_tensors="pt",
)

with torch.no_grad():
    outputs = model(**inputs)

report = tracker.report()
print(report)


# ============================================================
# MODE 2: context-manager style
# ============================================================

model = AutoModelForSequenceClassification.from_pretrained(model_name)

with FLOPpyTracker(
    run_name="hf_transformer_test_with",
) as tracker:

    tracker.start(model=model, tokenizer=base_tokenizer)

    inputs = tracker.tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**inputs)

    tracker.stop()

report = tracker.report()
print(report)
