from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Dict, Optional

from backends import create_backend
from ft_logging import create_logger
from training_hooks import TorchTrainingHooks


class Tracker(AbstractContextManager):
    """
    Tracker hook-only.

    Responsabilità:
    - inizializza backend e logger
    - aggancia gli hook FLOP dei layer tramite backend
    - aggancia gli hook di training (model/loss/optimizer) tramite TorchTrainingHooks
    - espone metriche finali
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

        # contatori extra
        self._preproc_ops: int = 0
        self._loss_flop: int = 0

        self._epoch_idx: int = 0

        # training hooks
        self._hooks: Optional[TorchTrainingHooks] = None

        # stato per debug / diagnostica
        self._last_model_output: Any = None
        self._last_backward_seen: bool = False
        self._last_optimizer_step_seen: bool = False

    # ------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------

    def __enter__(self):
        self.backend.start()
        return self

    def __exit__(self, exc_type, exc, tb):
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

    # ------------------------------------------------------------
    # Metriche
    # ------------------------------------------------------------

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
        # total_flop include già i FLOPs extra aggiunti via add_extra_flop 
        return float(self.total_flop + self.total_preproc_ops)

    # ------------------------------------------------------------
    # API per preprocessing / tokenizer
    # ------------------------------------------------------------

    def add_preproc_ops(self, ops: int) -> None:
        if ops is None:
            return
        v = int(ops)
        if v > 0:
            self._preproc_ops += v

    # ------------------------------------------------------------
    # API epoch (per aggiornare l'epoch dal training loop)
    # ------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        self._epoch_idx = int(epoch)
        if hasattr(self.backend, "set_epoch"):
            self.backend.set_epoch(self._epoch_idx)

    def log_epoch(self) -> None:
        """
        Log manuale per epoca.
        In modalità hook-only non possiamo sapere automaticamente quando finisce un'epoca,
        quindi questa funzione può essere richiamata dal training loop, se desiderato.
        """
        if self.logger is not None and hasattr(self.logger, "log_epoch"):
            self.logger.log_epoch(
                epoch=int(self._epoch_idx),
                flop=self.total_flop,
                cumulative_flop=self.total_flop,
            )

    # ------------------------------------------------------------
    # API hooks (Torch)
    # ------------------------------------------------------------

    def attach_torch_hooks(
        self,
        model,
        loss_fn=None,
        optimizer=None,
        enable_debug_print: bool = False,
    ) -> None:
        """
        Installa hook PyTorch per osservare training standard.
        """
        if self._hooks is not None:
            try:
                self._hooks.uninstall()
            except Exception:
                pass

        self._hooks = TorchTrainingHooks(self, enable_debug_print=enable_debug_print)
        self._hooks.install(model=model, loss_fn=loss_fn, optimizer=optimizer)

    # ------------------------------------------------------------
    # Utility per stima FLOP loss
    # ------------------------------------------------------------

    def _extract_preds(self, outputs: Any):
        """
        Estrae logits/preds da output Torch o HF.
        """
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

        # torch tensor diretto
        if isinstance(outputs, torch.Tensor):
            return outputs

        # tuple/list: primo tensor utile
        if isinstance(outputs, (tuple, list)):
            for o in outputs:
                if isinstance(o, torch.Tensor):
                    return o

        return None

    def _estimate_loss_flop(
        self,
        loss: Any,
        outputs: Any,
        targets: Any,
        extra: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Stima teorica dei FLOP della loss.

        Parametri:
        - loss: modulo loss oppure placeholder equivalente
        - outputs: preds/logits
        - targets: target tensor
        - extra: opzionale, può contenere loss_type

        loss_type supportati:
          "cross_entropy", "mse", "l1", "bce", "bce_logits", "kl"
        """
        try:
            import torch
            import torch.nn as nn
        except Exception:
            return 0

        preds = self._extract_preds(outputs)
        if not isinstance(preds, torch.Tensor):
            return 0

        # 1) prova a inferire loss_type da extra
        loss_type = None
        if extra and isinstance(extra, dict):
            lt = extra.get("loss_type", None)
            if isinstance(lt, str):
                loss_type = lt.lower().strip()

        # 2) se "loss" è proprio un modulo nn.*, inferiscilo da lì
        if loss_type is None and loss is not None:
            if isinstance(loss, nn.CrossEntropyLoss):
                loss_type = "cross_entropy"
            elif isinstance(loss, nn.MSELoss):
                loss_type = "mse"
            elif isinstance(loss, nn.L1Loss):
                loss_type = "l1"
            elif isinstance(loss, nn.BCELoss):
                loss_type = "bce"
            elif isinstance(loss, nn.BCEWithLogitsLoss):
                loss_type = "bce_logits"
            elif isinstance(loss, nn.KLDivLoss):
                loss_type = "kl"

        # 3) euristiche su target / preds
        if loss_type is None and isinstance(targets, torch.Tensor):
            if targets.dtype in (torch.int64, torch.int32, torch.int16, torch.int8):
                if preds.dim() >= 2 and preds.shape[-1] > 1:
                    loss_type = "cross_entropy"

            if loss_type is None and targets.dtype.is_floating_point:
                if preds.shape == targets.shape:
                    loss_type = "mse"

        # 4) formula FLOP
        if loss_type == "cross_entropy":
            if preds.dim() == 2:
                b, c = int(preds.shape[0]), int(preds.shape[1])
                return max(0, b * (3 * c + 2) + (b - 1))

            if preds.dim() == 3:
                b, t, c = int(preds.shape[0]), int(preds.shape[1]), int(preds.shape[2])
                n = b * t
                return max(0, n * (3 * c + 2) + (n - 1))

            return 0

        if loss_type == "mse":
            n = int(preds.numel())
            return max(0, 2 * n + (n - 1))

        if loss_type == "l1":
            n = int(preds.numel())
            return max(0, 2 * n + (n - 1))

        if loss_type == "bce":
            n = int(preds.numel())
            return max(0, 6 * n + (n - 1))

        if loss_type in ("bce_logits", "bcewithlogits"):
            n = int(preds.numel())
            return max(0, 10 * n + (n - 1))

        if loss_type == "kl":
            n = int(preds.numel())
            return max(0, 3 * n + (n - 1))

        return 0
