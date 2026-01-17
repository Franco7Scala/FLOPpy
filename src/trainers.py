from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

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
                    obs.on_after_loss(bc, loss_fn, outputs, yb)

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

def train_hf(
    *,
    model,
    dataloader,
    optimizer=None,
    device: Optional[str] = None,
    epochs: int = 1,
    observers: Optional[List[TrainingObserver]] = None,
    loss_kind: str = "auto",  # "auto" | "cross_entropy" | "none"
) -> None:
    """
    Trainer HF generico:
    - dataloader produce dict (input_ids, attention_mask, labels, ...)
    - se optimizer è None -> solo forward (inference)
    - se outputs.loss esiste e optimizer non è None -> backward + step
    - opzionale: notifica loss FLOP stimando CrossEntropy se loss_kind lo consente
    """
    observers = observers or []

    if device is not None:
        model.to(device)

    for obs in observers:
        obs.on_train_start({"backend": "hf"})

    # loss_fn "virtuale" per stimare FLOP della loss (quando HF calcola outputs.loss internamente)
    ce_loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for obs in observers:
            obs.on_epoch_start(epoch)

        for batch_idx, batch in enumerate(dataloader):
            # batch context
            batch_size = None
            if isinstance(batch, dict):
                # prova a stimare batch_size da input_ids
                ids = batch.get("input_ids", None)
                if hasattr(ids, "shape") and len(ids.shape) >= 1:
                    batch_size = int(ids.shape[0])

            bc = BatchContext(epoch=epoch, batch_idx=batch_idx, batch_size=batch_size)

            for obs in observers:
                obs.on_batch_start(bc)

            if device is not None and isinstance(batch, dict):
                batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(**batch) if isinstance(batch, dict) else model(batch)

            for obs in observers:
                obs.on_after_forward(bc, outputs)

            # training step se possibile
            loss = getattr(outputs, "loss", None)
            labels = batch.get("labels", None) if isinstance(batch, dict) else None

            if optimizer is not None and loss is not None:
                # NOTIFICA LOSS per conteggio FLOP
                # - "auto": usa CE se labels sono presenti e sono interi (classi)
                # - "cross_entropy": forza CE
                # - "none": non notifica loss FLOP
                if loss_kind != "none":
                    use_ce = False
                    if loss_kind == "cross_entropy":
                        use_ce = True
                    elif loss_kind == "auto":
                        # Heuristica: labels int64/long -> classification
                        if labels is not None and hasattr(labels, "dtype"):
                            use_ce = str(labels.dtype).endswith("int64") or "Long" in str(labels.dtype)

                    if use_ce and labels is not None:
                        logits = getattr(outputs, "logits", None)
                        if logits is not None:
                            for obs in observers:
                                obs.on_after_loss(bc, ce_loss_fn, logits, labels)

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
# HF GENERATIVE TRAINER 
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
    loss_kind: str = "none",  # spesso HF calcola CE internamente; per stimare, settare "cross_entropy"
) -> None:
    """
    Trainer per CausalLM (es. GPT2, LLaMA...)
    - se manca labels e create_labels_if_missing=True: labels = input_ids.clone()
    - se outputs.loss e optimizer: backward+step
    - opzionale: notifica FLOP loss (CrossEntropy) se loss_kind="cross_entropy"
    """
    observers = observers or []

    if device is not None:
        model.to(device)

    for obs in observers:
        obs.on_train_start({"backend": "hf_generative"})

    ce_loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for obs in observers:
            obs.on_epoch_start(epoch)

        for batch_idx, batch in enumerate(dataloader):
            if not isinstance(batch, dict):
                raise ValueError("train_hf_generative si aspetta batch dict (HF).")

            if create_labels_if_missing and "labels" not in batch and "input_ids" in batch:
                batch["labels"] = batch["input_ids"].clone()

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
                if loss_kind == "cross_entropy":
                    logits = getattr(outputs, "logits", None)
                    if logits is not None and labels is not None:
                        # per causal LM: logits shape (B, T, V), labels (B, T)
                        for obs in observers:
                            obs.on_after_loss(bc, ce_loss_fn, logits, labels)

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
# SKLEARN TRAINER (fit/predict/transform osservati)
# ------------------------------------------------------------

def train_sklearn(
    *,
    model,
    mode: str,
    X,
    y=None,
    observers: Optional[List[TrainingObserver]] = None,
) -> None:
    """
    Esegue chiamate sklearn con Observer.
    Il conteggio FLOP “vero” è nel backend SklearnBackend (wrapper di fit/predict/transform).
    Qui notifico solo la struttura del training (epoch/batch).
    """
    observers = observers or []

    for obs in observers:
        obs.on_train_start({"backend": "sklearn"})

    epoch = 0
    for obs in observers:
        obs.on_epoch_start(epoch)

    # ogni call = "batch"
    bc = BatchContext(epoch=epoch, batch_idx=0, batch_size=getattr(X, "shape", [None])[0] if hasattr(X, "shape") else None)
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
        # seconda call come secondo batch
        for obs in observers:
            obs.on_batch_end(bc)

        bc2 = BatchContext(epoch=epoch, batch_idx=1, batch_size=getattr(X, "shape", [None])[0] if hasattr(X, "shape") else None)
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
