from __future__ import annotations
from contextlib import AbstractContextManager
from typing import Any, Dict, Optional

from backends import create_backend
from ft_logging import create_logger
from observers import TrainingObserver, BatchContext


class Tracker(AbstractContextManager, TrainingObserver):
    """
    Tracker Observer:
    - Instanzia backend e logger
    - Aggancia hook FLOPs modello
    - Osserva eventi del training per includere costi extra (loss, tokenizer, ecc.)
    - Espone metriche finali
    """

    def __init__(
        self,
        model,
        backend: str = "auto",
        log_per_batch: bool = False,
        log_per_epoch: bool = False,
        export_path: str | None = None,
        use_wandb: bool = False,
        wandb_project: str | None = None,
        wandb_token: str | None = None,
        run_name: str | None = None,
    ):
        self.logger = create_logger(
            log_per_batch=log_per_batch,
            log_per_epoch=log_per_epoch,
            export_path=export_path,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_token=wandb_token,
            run_name=run_name,
        )
        self.backend = create_backend(model, backend, logger=self.logger)

        # contatori extra (fuori dagli hook)
        self._preproc_ops: int = 0
        self._loss_flop: int = 0

        self._epoch_idx: int = 0

    # ---------------- Context Manager ---------------- #

    def __enter__(self):
        self.backend.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.backend.stop()
        if self.logger is not None:
            self.logger.close()
        return False

    # ---------------- Metriche ---------------- #

    @property
    def total_flop(self) -> int:
        return self.backend.get_total_flop()

    @property
    def total_preproc_ops(self) -> int:
        return int(self._preproc_ops)

    @property
    def total_loss_flop(self) -> int:
        return int(self._loss_flop)

    @property
    def total_operations(self) -> float:
        # aggregato (modello + tokenizer/preproc)
        return float(self.total_flop + self.total_preproc_ops)

    # ---------------- API per preproc/tokenizer ---------------- #

    def add_preproc_ops(self, ops: int) -> None:
        if ops is None:
            return
        v = int(ops)
        if v > 0:
            self._preproc_ops += v

    # ---------------- Observer callbacks ---------------- #

    def on_train_start(self, ctx: Dict[str, Any] | None = None) -> None:
        return

    def on_epoch_start(self, epoch: int) -> None:
        self._epoch_idx = int(epoch)
        if hasattr(self.backend, "set_epoch"):
            self.backend.set_epoch(self._epoch_idx)

    def on_batch_start(self, bc: BatchContext) -> None:
        return

    def on_after_forward(self, bc: BatchContext, outputs: Any) -> None:
        return

    def on_after_loss(self, bc: BatchContext, loss: Any, outputs: Any, targets: Any) -> None:
        # stima FLOPs loss e aggiungi al batch corrente
        loss_flops = self._estimate_loss_flop(loss, outputs, targets)
        if loss_flops > 0:
            self._loss_flop += int(loss_flops)
            if hasattr(self.backend, "add_extra_flop"):
                self.backend.add_extra_flop(int(loss_flops))

    def on_after_backward(self, bc: BatchContext) -> None:
        return

    def on_after_step(self, bc: BatchContext) -> None:
        return

    def on_batch_end(self, bc: BatchContext) -> None:
        return

    def on_epoch_end(self, epoch: int) -> None:
        # log per epoch (se abilitato)
        if self.logger is not None and hasattr(self.logger, "log_epoch"):
            self.logger.log_epoch(
                epoch=int(epoch),
                flop=self.total_flop,
                cumulative_flop=self.total_flop,
            )

    def on_train_end(self, ctx: Dict[str, Any] | None = None) -> None:
        return

    # ---------------- Loss FLOPs (stima) ---------------- #

    def _estimate_loss_flop(self, loss_fn, preds, targets) -> int:
        """
        Stima teorica dei FLOPs della loss (forward).
        Usa euristiche robuste; se non può stimare, restituisce 0.
        """
        try:
            import torch
            import torch.nn as nn
        except Exception:
            return 0

        if loss_fn is None or preds is None or not hasattr(preds, "numel"):
            return 0

        N = int(preds.numel())

        if isinstance(loss_fn, nn.MSELoss):
            return max(0, 2 * N + (N - 1))
        if isinstance(loss_fn, nn.L1Loss):
            return max(0, 2 * N + (N - 1))

        if isinstance(loss_fn, nn.NLLLoss):
            B = int(preds.shape[0]) if hasattr(preds, "shape") and preds.dim() > 0 else 1
            return max(0, B + (B - 1))

        if isinstance(loss_fn, nn.CrossEntropyLoss):
            B = int(preds.shape[0]) if hasattr(preds, "shape") and preds.dim() > 0 else 1
            C = int(preds.shape[-1]) if hasattr(preds, "shape") and preds.dim() > 0 else 1
            return max(0, B * (4 * C) + B + (B - 1))

        if isinstance(loss_fn, nn.KLDivLoss):
            return max(0, 3 * N + (N - 1))

        if isinstance(loss_fn, nn.BCELoss):
            return max(0, 6 * N + (N - 1))

        if isinstance(loss_fn, nn.BCEWithLogitsLoss):
            return max(0, 10 * N + (N - 1))

        if isinstance(loss_fn, (nn.TripletMarginLoss, nn.TripletMarginWithDistanceLoss)):
            return max(0, 5 * N)

        return 0
