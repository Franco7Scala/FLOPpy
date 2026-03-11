from __future__ import annotations
from contextlib import AbstractContextManager
from typing import Any, Dict, Optional
from backends import create_backend
from ft_logging import create_logger
from training_hooks import TorchTrainingHooks

class Tracker(AbstractContextManager):
    def __init__(
        self,
        model,
        backend: str = "auto",
        export_path: Optional[str] = None,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_token: Optional[str] = None,
        run_name: Optional[str] = None,
    ):
        self.logger = create_logger(
            export_path=export_path,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_token=wandb_token,
            run_name=run_name,
        )

        self.backend = create_backend(model, backend, logger=self.logger)

        self._preproc_ops: int = 0
        self._loss_forward_flop: int = 0
        self._loss_backward_flop: int = 0
        self._optimizer_flop: int = 0

        self._hooks: Optional[TorchTrainingHooks] = None

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

        if self.logger is not None and hasattr(self.logger, "log_summary"):
            try:
                self.logger.log_summary(self.build_summary_dict())
            except Exception:
                pass

        if self.logger is not None:
            self.logger.close()

        return False

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------

    @property
    def total_model_flop(self) -> int:
        return self.backend.get_total_flop()

    @property
    def total_preproc_ops(self) -> int:
        return int(self._preproc_ops)

    @property
    def total_loss_forward_flop(self) -> int:
        return int(self._loss_forward_flop)

    @property
    def total_loss_backward_flop(self) -> int:
        return int(self._loss_backward_flop)

    @property
    def total_optimizer_flop(self) -> int:
        return int(self._optimizer_flop)

    @property
    def total_overall_flop(self) -> int:
        return (
            self.total_model_flop
            + self.total_loss_forward_flop
            + self.total_loss_backward_flop
            + self.total_optimizer_flop
        )

    # ------------------------------------------------------------
    # API preproc
    # ------------------------------------------------------------

    def add_preproc_ops(self, ops: int) -> None:
        if ops is None:
            return
        v = int(ops)
        if v > 0:
            self._preproc_ops += v

    # ------------------------------------------------------------
    # Hooks API
    # ------------------------------------------------------------

    def attach_torch_hooks(
        self,
        model,
        loss_fn=None,
        optimizer=None,
        enable_debug_print: bool = False,
    ) -> None:
        if self._hooks is not None:
            try:
                self._hooks.uninstall()
            except Exception:
                pass

        self._hooks = TorchTrainingHooks(self, enable_debug_print=enable_debug_print)
        self._hooks.install(model=model, loss_fn=loss_fn, optimizer=optimizer)

    # ------------------------------------------------------------
    # Final summary 
    # ------------------------------------------------------------

    def build_summary_dict(self) -> Dict[str, int]:
        return {
            "total_model_flop": self.total_model_flop,
            "total_optimizer_flop": self.total_optimizer_flop,
            "total_loss_forward_flop": self.total_loss_forward_flop,
            "total_loss_backward_flop": self.total_loss_backward_flop,
            "total_overall_flop": self.total_overall_flop,
        }

    # ------------------------------------------------------------
    # Utility loss FLOP
    # ------------------------------------------------------------

    def _extract_preds(self, outputs: Any):
        try:
            import torch
        except Exception:
            return None

        if outputs is None:
            return None

        logits = getattr(outputs, "logits", None)
        if isinstance(logits, torch.Tensor):
            return logits

        if isinstance(outputs, torch.Tensor):
            return outputs

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
        try:
            import torch
            import torch.nn as nn
        except Exception:
            return 0

        preds = self._extract_preds(outputs)
        if not isinstance(preds, torch.Tensor):
            return 0

        loss_type = None
        if extra and isinstance(extra, dict):
            lt = extra.get("loss_type", None)
            if isinstance(lt, str):
                loss_type = lt.lower().strip()

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

        if loss_type is None and isinstance(targets, torch.Tensor):
            if targets.dtype in (torch.int64, torch.int32, torch.int16, torch.int8):
                if preds.dim() >= 2 and preds.shape[-1] > 1:
                    loss_type = "cross_entropy"
            if loss_type is None and targets.dtype.is_floating_point:
                if preds.shape == targets.shape:
                    loss_type = "mse"

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

    def _estimate_loss_backward_flop(
        self,
        loss: Any,
        outputs: Any,
        targets: Any,
        extra: Optional[Dict[str, Any]] = None,
    ) -> int:
        return self._estimate_loss_flop(loss=loss, outputs=outputs, targets=targets, extra=extra)

    def _estimate_optimizer_flop(self, optimizer) -> int:
        try:
            import torch.optim as optim
        except Exception:
            return 0

        n_params = 0
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p is not None and hasattr(p, "numel"):
                    n_params += int(p.numel())

        if isinstance(optimizer, optim.SGD):
            return 2 * n_params

        if isinstance(optimizer, (optim.Adam, optim.AdamW)):
            return 10 * n_params

        if isinstance(optimizer, optim.RMSprop):
            return 8 * n_params

        return 4 * n_params
