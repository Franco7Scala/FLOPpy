import os
import sys

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from flop_tracker import FlopTracker
from trainers import train_hf_generative

try:
    from tokenizer_ops import TokenizerWithOps
except Exception:
    TokenizerWithOps = None


SENTENCES = [
    "Hello, this is a small generative test.",
    "We are measuring FLOPs for a causal language model.",
    "The tracker is observing the training loop externally.",
    "This sentence is just for batch variety.",
]


def collate_fn(batch_texts, tokenizer, max_length=64, tracker=None):
    enc = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    # CausalLM: labels = input_ids
    enc["labels"] = enc["input_ids"].clone()

    # opzionale: ops tokenizer
    if tracker is not None and hasattr(tokenizer, "last_ops"):
        try:
            tracker.add_preproc_ops(int(tokenizer.last_ops))
        except Exception:
            pass

    return enc


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = "distilgpt2"
    base_tokenizer = AutoTokenizer.from_pretrained(model_name)
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token

    tokenizer = TokenizerWithOps(base_tokenizer) if TokenizerWithOps is not None else base_tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)

    tracker_holder = {"tr": None}

    def _collate(batch):
        return collate_fn(batch, tokenizer, tracker=tracker_holder["tr"])

    loader = DataLoader(SENTENCES, batch_size=2, shuffle=True, collate_fn=_collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    ft = FlopTracker(run_name="hf_distilgpt2_generative_observer", print_summary=True, print_hardware=True)

    def train_fn_wrapped(*, model, dataloader, optimizer, device=None, epochs=1, observers=None):
        observers = observers or []
        if observers:
            tracker_holder["tr"] = observers[0]
        return train_hf_generative(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            epochs=epochs,
            observers=observers,
            create_labels_if_missing=True,
        )

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
        export_path="hf_distilgpt2_generative_flop.csv",
        use_wandb=False,
    )


if __name__ == "__main__":
    main()
