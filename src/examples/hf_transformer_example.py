import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from floppy_tracker import FLOPpyTracker
from tokenizer_ops import wrap_tokenizer


def main():
    model_name = "distilbert-base-uncased"

    base_tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    tracker = FLOPpyTracker(
        run_name="hf_transformer_test",
        print_summary=True,
    )

    tokenizer = wrap_tokenizer(base_tokenizer, tracker=tracker._tracker if hasattr(tracker, "_tracker") else None)

    with tracker.run(
        model=model,
        export_path="hf_transformer_test.csv",
    ):
        tokenizer = wrap_tokenizer(base_tokenizer, tracker=tracker._tracker)
        texts = [
            "FLOP estimation is important for green AI.",
            "Tracking compute is useful for research.",
        ]
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():
            model(**inputs)

    rep = tracker.report

    assert rep.model_flop > 0


if __name__ == "__main__":
    main()
