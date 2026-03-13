import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from floppy_tracker import FLOPpyTracker


model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(model_name)

texts = [
    "FLOP estimation is important for green AI.",
    "Tracking compute helps improve model efficiency."
]

tracker = FLOPpyTracker(
    run_name="hf_transformer_example",
    print_summary=True
)

with tracker.run(
    model=model,
    export_path="hf_transformer_example.csv"
):

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
