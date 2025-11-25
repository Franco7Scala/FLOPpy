import os
import sys
import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from flop_tracker import FlopTracker


def collate_fn(batch, tokenizer, max_length=128):
    texts = [ex["sentence"] for ex in batch]
    labels = [ex["label"] for ex in batch]
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc["labels"] = torch.tensor(labels)
    return enc


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Dataset
    ds = load_dataset("glue", "sst2", split="train[:1%]")

    loader = DataLoader(
        ds,
        batch_size=16,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    ft = FlopTracker(run_name="hf_distilbert_sst2").hf_bind(
        model=model,
        dataloader=loader,
        optimizer=optimizer,
        device=device,
        epochs=1,
        log_per_batch=True,
        log_per_epoch=True,
        export_path="hf_distilbert_flop.csv",
        use_wandb=False,
    )

    print("Raw FLOP:", ft.raw_flop)
    print("Total FLOP:", ft.total_flop)


if __name__ == "__main__":
    main()
