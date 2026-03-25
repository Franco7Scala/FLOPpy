import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from floppy import FLOPpyTracker

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "distilbert-base-uncased"
texts = [
    "FLOP estimation is important for green AI.",
    "Tracking compute helps improve model efficiency.",
]

base_tokenizer = AutoTokenizer.from_pretrained(model_name)

# ============================================================
# MODE 1
# ============================================================

model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

tracker = FLOPpyTracker(run_name="hf_transformer_test")
tracker.run(model=model, tokenizer=base_tokenizer)

inputs = tracker.tokenizer(
    texts,
    padding=True,
    truncation=True,
    return_tensors="pt",
)
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

report = tracker.report()
print_report_metrics(report)

# ============================================================
# MODE 2: context-manager style
# ============================================================

model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

with FLOPpyTracker(run_name="hf_transformer_test_with") as tracker:
    tracker.start(model=model, tokenizer=base_tokenizer)

    inputs = tracker.tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    tracker.stop()

report = tracker.report()
print_report_metrics(report)
