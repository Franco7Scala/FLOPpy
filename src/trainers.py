from __future__ import annotations

from typing import Any, Dict, List, Optional

from observers import TrainingObserver, BatchContext


# ------------------------------------------------------------
# TORCH TRAINER
# ------------------------------------------------------------

def train_torch(
    *,
    model,
    optimizer,
    loss_fn,
    train_loader,
    device: Optional[str] = None,
    epochs: int = 1,
    observers: Optional[List[TrainingObserver]] = None,
) -> None:
    """
    Trainer PyTorch (training esterno).
    Import di torch LAZY: sklearn può funzionare senza torch installato.
    """
    import torch  # lazy import

    observers = observers or []

    if device is not None:
        model.to(device)

    for obs in observers:
        obs.on_train_start({"backend": "torch"})

    for epoch in range(epochs):
        for obs in observers:
            obs.on_epoch_start(epoch)

        for batch_idx, (xb, yb) in enumerate(train_loader):
            if device is not None:
                xb, yb = xb.to(device), yb.to(device)

            bc = BatchContext(
                epoch=epoch,
                batch_idx=batch_idx,
                batch_size=int(xb.shape[0]) if hasattr(xb, "shape") and xb is not None else None,
            )

            for obs in observers:
                obs.on_batch_start(bc)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(xb)

            for obs in observers:
                obs.on_after_forward(bc, outputs)

            if loss_fn is not None:
                loss = loss_fn(outputs, yb)

                for obs in observers:
                    obs.on_after_loss(bc, loss, outputs, yb)

                loss.backward()
                for obs in observers:
                    obs.on_after_backward(bc)

                optimizer.step()
                for obs in observers:
                    obs.on_after_step(bc)

            for obs in observers:
                obs.on_batch_end(bc)

        for obs in observers:
            obs.on_epoch_end(epoch)

    for obs in observers:
        obs.on_train_end({"backend": "torch"})


# ------------------------------------------------------------
# HF TRAINER (classification / seq2seq / generic)
# ------------------------------------------------------------

def _normalize_hf_batch(batch: Any) -> Any:
    """
    Normalizza i batch HF in modo robusto:
    - se batch è BatchEncoding -> dict(batch)
    - se batch è dict con liste/tuple -> torch.tensor(...)
    """
    try:
        import torch
    except Exception:
        torch = None

    # BatchEncoding -> dict
    try:
        from transformers.tokenization_utils_base import BatchEncoding
    except Exception:
        BatchEncoding = None

    if BatchEncoding is not None and isinstance(batch, BatchEncoding):
        batch = dict(batch)

    if isinstance(batch, dict) and torch is not None:
        for k, v in list(batch.items()):
            if hasattr(v, "to"):
                continue
            # liste/tuple -> tensor
            if isinstance(v, (list, tuple)):
                try:
                    batch[k] = torch.tensor(v)
                except Exception:
                    pass

    return batch


def train_hf(
    *,
    model,
    dataloader,
    optimizer=None,
    device: Optional[str] = None,
    epochs: int = 1,
    observers: Optional[List[TrainingObserver]] = None,
) -> None:
    """
    Trainer HF generico:
    - dataloader produce dict (input_ids, attention_mask, labels, ...)
    - se optimizer è None -> solo forward (inference)
    - se outputs.loss esiste e optimizer non è None -> backward + step
    """
    import torch  # lazy import 

    observers = observers or []

    if device is not None:
        model.to(device)

    for obs in observers:
        obs.on_train_start({"backend": "hf"})

    for epoch in range(epochs):
        for obs in observers:
            obs.on_epoch_start(epoch)

        for batch_idx, batch in enumerate(dataloader):
            batch = _normalize_hf_batch(batch)

            # batch_size robusto
            batch_size = None
            if isinstance(batch, dict):
                ids = batch.get("input_ids", None)
                if hasattr(ids, "shape") and len(ids.shape) >= 1:
                    batch_size = int(ids.shape[0])

            bc = BatchContext(epoch=epoch, batch_idx=batch_idx, batch_size=batch_size)

            for obs in observers:
                obs.on_batch_start(bc)

            # move to device
            if device is not None and isinstance(batch, dict):
                batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(**batch) if isinstance(batch, dict) else model(batch)

            for obs in observers:
                obs.on_after_forward(bc, outputs)

            loss = getattr(outputs, "loss", None)
            labels = batch.get("labels", None) if isinstance(batch, dict) else None

            if optimizer is not None and loss is not None:
                for obs in observers:
                    obs.on_after_loss(bc, loss, outputs, labels)

                loss.backward()
                for obs in observers:
                    obs.on_after_backward(bc)

                optimizer.step()
                for obs in observers:
                    obs.on_after_step(bc)

            for obs in observers:
                obs.on_batch_end(bc)

        for obs in observers:
            obs.on_epoch_end(epoch)

    for obs in observers:
        obs.on_train_end({"backend": "hf"})


# ------------------------------------------------------------
# HF GENERATIVE TRAINER (Causal LM)
# ------------------------------------------------------------

def train_hf_generative(
    *,
    model,
    dataloader,
    optimizer=None,
    device: Optional[str] = None,
    epochs: int = 1,
    observers: Optional[List[TrainingObserver]] = None,
    create_labels_if_missing: bool = True,
) -> None:
    """
    Trainer per CausalLM:
    - se manca labels e create_labels_if_missing=True: labels = input_ids.clone()
    - se outputs.loss e optimizer: backward+step
    """
    import torch  # lazy import

    observers = observers or []

    if device is not None:
        model.to(device)

    for obs in observers:
        obs.on_train_start({"backend": "hf_generative"})

    for epoch in range(epochs):
        for obs in observers:
            obs.on_epoch_start(epoch)

        for batch_idx, batch in enumerate(dataloader):
            batch = _normalize_hf_batch(batch)

            if not isinstance(batch, dict):
                raise ValueError("train_hf_generative si aspetta batch dict (HF).")

            if create_labels_if_missing and "labels" not in batch and "input_ids" in batch:
                try:
                    batch["labels"] = batch["input_ids"].clone()
                except Exception:
                    pass

            ids = batch.get("input_ids", None)
            batch_size = int(ids.shape[0]) if hasattr(ids, "shape") and len(ids.shape) >= 1 else None
            bc = BatchContext(epoch=epoch, batch_idx=batch_idx, batch_size=batch_size)

            for obs in observers:
                obs.on_batch_start(bc)

            if device is not None:
                batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(**batch)

            for obs in observers:
                obs.on_after_forward(bc, outputs)

            loss = getattr(outputs, "loss", None)
            labels = batch.get("labels", None)

            if optimizer is not None and loss is not None:
                for obs in observers:
                    obs.on_after_loss(bc, loss, outputs, labels)

                loss.backward()
                for obs in observers:
                    obs.on_after_backward(bc)

                optimizer.step()
                for obs in observers:
                    obs.on_after_step(bc)

            for obs in observers:
                obs.on_batch_end(bc)

        for obs in observers:
            obs.on_epoch_end(epoch)

    for obs in observers:
        obs.on_train_end({"backend": "hf_generative"})


# ------------------------------------------------------------
# SKLEARN TRAINER
# ------------------------------------------------------------

def train_sklearn(
    *,
    model,
    mode: str,
    X,
    y=None,
    observers: Optional[List[TrainingObserver]] = None,
) -> None:
    observers = observers or []

    for obs in observers:
        obs.on_train_start({"backend": "sklearn"})

    epoch = 0
    for obs in observers:
        obs.on_epoch_start(epoch)

    bs = getattr(X, "shape", [None])[0] if hasattr(X, "shape") else None
    bc = BatchContext(epoch=epoch, batch_idx=0, batch_size=int(bs) if bs is not None else None)

    for obs in observers:
        obs.on_batch_start(bc)

    if mode == "fit":
        model.fit(X, y)
    elif mode == "predict":
        _ = model.predict(X)
    elif mode == "predict_proba":
        _ = model.predict_proba(X)
    elif mode == "transform":
        _ = model.transform(X)
    elif mode == "fit_predict":
        model.fit(X, y)

        for obs in observers:
            obs.on_batch_end(bc)

        bc2 = BatchContext(epoch=epoch, batch_idx=1, batch_size=int(bs) if bs is not None else None)
        for obs in observers:
            obs.on_batch_start(bc2)

        _ = model.predict(X)

        for obs in observers:
            obs.on_batch_end(bc2)

        for obs in observers:
            obs.on_epoch_end(epoch)
        for obs in observers:
            obs.on_train_end({"backend": "sklearn"})
        return
    else:
        raise ValueError(f"Modalità sklearn non supportata: {mode}")

    for obs in observers:
        obs.on_batch_end(bc)

    for obs in observers:
        obs.on_epoch_end(epoch)

    for obs in observers:
        obs.on_train_end({"backend": "sklearn"})
