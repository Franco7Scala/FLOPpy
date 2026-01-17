import os
import sys

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from flop_tracker import FlopTracker
from trainers import train_hf

# (opzionale) wrapper tokenizer ops
try:
    from tokenizer_ops import TokenizerWithOps
except Exception:
    TokenizerWithOps = None


def collate_fn(batch, tokenizer, max_length=128, tracker=None):
    texts = [ex["sentence"] for ex in batch]
    labels = [ex["label"] for ex in batch]

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    if tracker is not None and hasattr(tokenizer, "last_ops"):
        try:
            tracker.add_preproc_ops(int(tokenizer.last_ops))
        except Exception:
            pass

    enc["labels"] = torch.tensor(labels)
    return enc


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = "distilbert-base-uncased"

    base_tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer = TokenizerWithOps(base_tokenizer) if TokenizerWithOps is not None else base_tokenizer

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    ds = load_dataset("glue", "sst2", split="train[:1%]")

    tracker_holder = {"tr": None}

    def _collate(batch):
        return collate_fn(batch, tokenizer, tracker=tracker_holder["tr"])

    loader = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=_collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    ft = FlopTracker(run_name="hf_distilbert_sst2_observer", print_summary=True, print_hardware=True)

    def train_fn_wrapped(*, model, dataloader, optimizer, device=None, epochs=1, observers=None):
        observers = observers or []
        if observers:
            tracker_holder["tr"] = observers[0]
        return train_hf(model=model, dataloader=dataloader, optimizer=optimizer, device=device, epochs=epochs, observers=observers)

    ft.run(
        model=model,
        backend="hf",
        train_fn=train_fn_wrapped,
        train_kwargs={
            "model": model,
            "dataloader": loader,
            "optimizer": optimizer,
            "device": device,
            "epochs": 1,
        },
        log_per_batch=True,
        log_per_epoch=True,
        export_path="hf_distilbert_sst2_flop.csv",
        use_wandb=False,
    )


if __name__ == "__main__":
    main()
