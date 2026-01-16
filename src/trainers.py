from __future__ import annotations
from typing import List, Optional
import torch

from observers import TrainingObserver, BatchContext


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

            optimizer.zero_grad()
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
