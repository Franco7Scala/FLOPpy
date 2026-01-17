import os
import sys
import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from flop_tracker import FlopTracker


def collate_fn(batch, tokenizer, max_length=64):
    texts = [ex["text"] for ex in batch]
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc["labels"] = enc["input_ids"].clone()
    return enc


def train_hf_causal(*, model, dataloader, optimizer=None, device=None, epochs=1, observers=None):
    observers = observers or []

    if device is not None:
        model.to(device)

    for obs in observers:
        obs.on_train_start({"backend": "hf_generative"})

    for epoch in range(epochs):
        for obs in observers:
            obs.on_epoch_start(epoch)

        for batch_idx, batch in enumerate(dataloader):
            for obs in observers:
                obs.on_batch_start(type("BC", (), {"epoch": epoch, "batch_idx": batch_idx, "batch_size": None}))

            if device is not None:
                batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}

            if optimizer is not None:
                optimizer.zero_grad()

            outputs = model(**batch)

            for obs in observers:
                obs.on_after_forward(type("BC", (), {"epoch": epoch, "batch_idx": batch_idx, "batch_size": None}), outputs)

            loss = getattr(outputs, "loss", None)
            if loss is not None and optimizer is not None:
                for obs in observers:
                    obs.on_after_loss(type("BC", (), {"epoch": epoch, "batch_idx": batch_idx, "batch_size": None}), None, outputs, batch.get("labels"))

                loss.backward()
                for obs in observers:
                    obs.on_after_backward(type("BC", (), {"epoch": epoch, "batch_idx": batch_idx, "batch_size": None}))

                optimizer.step()
                for obs in observers:
                    obs.on_after_step(type("BC", (), {"epoch": epoch, "batch_idx": batch_idx, "batch_size": None}))

            for obs in observers:
                obs.on_batch_end(type("BC", (), {"epoch": epoch, "batch_idx": batch_idx, "batch_size": None}))

        for obs in observers:
            obs.on_epoch_end(epoch)

    for obs in observers:
        obs.on_train_end({"backend": "hf_generative"})


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")

    loader = DataLoader(
        ds,
        batch_size=8,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    ft = FlopTracker(run_name="hf_distilgpt2_facade", print_summary=True).run(
        model=model,
        backend="hf",
        train_fn=train_hf_causal,
        train_kwargs=dict(
            model=model,
            dataloader=loader,
            optimizer=optimizer,
            device=device,
            epochs=1,
        ),
        log_per_batch=True,
        log_per_epoch=True,
        export_path="hf_distilgpt2_facade.csv",
        use_wandb=False,
    )

    print("REPORT:", ft.report)

    prompt = "Today I learned that"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out_ids = model.to(device).generate(**inputs, max_new_tokens=30)
    print("\nGenerated text:\n", tokenizer.decode(out_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
