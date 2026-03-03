from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Dict, Optional

from backends import create_backend
from ft_logging import create_logger
from observers import TrainingObserver, BatchContext

from training_hooks import TorchTrainingHooks


class Tracker(AbstractContextManager, TrainingObserver):
    """
    Tracker Observer:
    - Instanzia backend e logger
    - Aggancia hook FLOP modello (backend)
    - Osserva eventi del training (Observer) per includere costi extra (loss, tokenizer, ecc.)
    - Opzionalmente usa TorchTrainingHooks per "osservare" il training via hook PyTorch
      senza richiedere trainers observer-aware
    - Espone metriche finali
    """

    def __init__(
        self,
        model,
        backend: str = "auto",
        log_per_batch: bool = False,
        log_per_epoch: bool = False,
        export_path: Optional[str] = None,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_token: Optional[str] = None,
        run_name: Optional[str] = None,
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

        # contatori extra (fuori dagli hook di layer)
        self._preproc_ops: int = 0
        self._loss_flop: int = 0

        self._epoch_idx: int = 0

        # training hooks (loss forward/backward, optimizer step, ecc.)
        self._hooks: Optional[TorchTrainingHooks] = None

    # ---------------- Context Manager ---------------- #

    def __enter__(self):
        self.backend.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        # rimuove hook training se installati
        try:
            if self._hooks is not None:
                self._hooks.uninstall()
                self._hooks = None
        except Exception:
            pass

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
        # NON aggregare loss qui, perché la loss viene già aggiunta al backend tramite add_extra_flop
        # -> quindi total_flop include già "modello + extra".
        return float(self.total_flop + self.total_preproc_ops)

    # ---------------- API per preproc/tokenizer ---------------- #

    def add_preproc_ops(self, ops: int) -> None:
        if ops is None:
            return
        v = int(ops)
        if v > 0:
            self._preproc_ops += v

    # ---------------- API HOOKS (Torch) ---------------- #

    def attach_torch_hooks(
        self,
        *,
        model,
        loss_fn=None,
        optimizer=None,
        enable_debug_print: bool = False,
    ) -> None:
        """
        Installa hook PyTorch per osservare training senza trainers observer-aware.

        - model: serve per register_forward_hook
        - loss_fn: serve per hook forward/backward sulla loss
        - optimizer: serve per hook post-step (se supportato dalla versione torch)
        """
        self._hooks = TorchTrainingHooks(self, enable_debug_print=enable_debug_print)
        self._hooks.install(model=model, loss_fn=loss_fn, optimizer=optimizer)

    # ---------------- Observer callbacks ---------------- #

    def on_train_start(self, ctx: Optional[Dict[str, Any]] = None) -> None:
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
        """
        Modalità Observer-classica (trainers observer-aware).
        In modalità HOOKS invece, la loss viene contabilizzata da training_hooks.py.
        """
        loss_flop = self._estimate_loss_flop(loss=loss, outputs=outputs, targets=targets, extra=bc.extra)
        if loss_flop > 0:
            self._loss_flop += int(loss_flop)
            if hasattr(self.backend, "add_extra_flop"):
                self.backend.add_extra_flop(int(loss_flop))

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

    def on_train_end(self, ctx: Optional[Dict[str, Any]] = None) -> None:
        return

    # ---------------- Stima Loss FLOP ---------------- #

    def _extract_preds(self, outputs: Any):
        """Estrae logits/preds da output Torch o HF."""
        try:
            import torch
        except Exception:
            return None

        if outputs is None:
            return None

        # HF ModelOutput: .logits
        logits = getattr(outputs, "logits", None)
        if isinstance(logits, torch.Tensor):
            return logits

        # torch: output tensor diretto
        if isinstance(outputs, torch.Tensor):
            return outputs

        # tuple/list: primo tensor
        if isinstance(outputs, (tuple, list)):
            for o in outputs:
                if isinstance(o, torch.Tensor):
                    return o

        return None

    def _estimate_loss_flop(self, loss: Any, outputs: Any, targets: Any, extra: Optional[Dict[str, Any]]) -> int:
        """
        Stima teorica FLOP della loss (solo forward), usando:
        - extra["loss_type"] se presente (consigliato)
        - altrimenti euristiche su preds/targets

        loss_type supportati:
          "cross_entropy", "mse", "l1", "bce", "bce_logits", "kl"
        """
        try:
            import torch
        except Exception:
            return 0

        preds = self._extract_preds(outputs)
        if not isinstance(preds, torch.Tensor):
            return 0

        # 1) loss_type esplicito (opzionale)
        loss_type = None
        if extra and isinstance(extra, dict):
            lt = extra.get("loss_type", None)
            if isinstance(lt, str):
                loss_type = lt.lower().strip()

        # 2) euristiche se non specificato
        if loss_type is None:
            if isinstance(targets, torch.Tensor):
                # Classification: targets int/long e preds ha classe dimension (.., C)
                if targets.dtype in (torch.int64, torch.int32, torch.int16, torch.int8):
                    if preds.dim() >= 2 and preds.shape[-1] > 1:
                        loss_type = "cross_entropy"
                # Regression-like: targets float e shape compatibile
                if loss_type is None and targets.dtype.is_floating_point:
                    if preds.shape == targets.shape:
                        loss_type = "mse"

        # 3) calcolo FLOP per tipo
        if loss_type == "cross_entropy":
            # preds: (B, C) o (B, T, C). targets: (B) o (B, T)
            if preds.dim() == 2:
                B, C = int(preds.shape[0]), int(preds.shape[1])
                return max(0, B * (3 * C + 2) + (B - 1))
            if preds.dim() == 3:
                B, T, C = int(preds.shape[0]), int(preds.shape[1]), int(preds.shape[2])
                n = B * T
                return max(0, n * (3 * C + 2) + (n - 1))
            return 0

        if loss_type == "mse":
            N = int(preds.numel())
            return max(0, 2 * N + (N - 1))

        if loss_type == "l1":
            N = int(preds.numel())
            return max(0, 2 * N + (N - 1))

        if loss_type == "bce":
            N = int(preds.numel())
            return max(0, 6 * N + (N - 1))

        if loss_type in ("bce_logits", "bcewithlogits"):
            N = int(preds.numel())
            return max(0, 10 * N + (N - 1))

        if loss_type == "kl":
            N = int(preds.numel())
            return max(0, 3 * N + (N - 1))

        return 0
