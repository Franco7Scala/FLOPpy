from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Optional
from backends import create_backend
from ft_logging import create_logger
from training_hooks import TorchTrainingHooks


class Tracker(AbstractContextManager):

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
