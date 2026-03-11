from __future__ import annotations
from contextlib import AbstractContextManager
from typing import Any, Dict, Optional
from backends import create_backend
from ft_logging import create_logger
from training_hooks import TorchTrainingHooks

class Tracker(AbstractContextManager):
    """
    Tracker hook-only.

    Responsibilities:
    - Initialize backend e logger
    - Activate FLOP count of the model by backend
    - Activate the training hook (loss/optimizer) by TorchTrainingHooks
    - Maintains the final global counters
    - Generates the final summary for loggers and reports
    """

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

        # Aggregate counters
        self._preproc_ops: int = 0
        self._loss_forward_flop: int = 0
        self._loss_backward_flop: int = 0
        self._optimizer_flop: int = 0

        # training hooks
        self._hooks: Optional[TorchTrainingHooks] = None

        # Internal state for debug / future extensibility
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
    # Final metrics
    # ------------------------------------------------------------

    @property
    def total_model_flop(self) -> int:
        return int(self.backend.get_total_flop())

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
    # API preprocessing / tokenizer
    # ------------------------------------------------------------

    def add_preproc_ops(self, ops: int) -> None:
        if ops is None:
            return
        value = int(ops)
        if value > 0:
            self._preproc_ops += value

    # ------------------------------------------------------------
    # API hooks (torch)
    # ------------------------------------------------------------

    def attach_torch_hooks(
        self,
        model,
        loss_fn=None,
        optimizer=None,
        enable_debug_print: bool = False,
    ) -> None:
        """
        Installs PyTorch hooks for loss and optimizer.
        
        Note:
        - Model FLOP are counted by the backend.
        - Loss and optimizer FLOP are counted dynamically by TorchTrainingHooks via UniversalFlopCounter.
        """
        if self._hooks is not None:
            try:
                self._hooks.uninstall()
            except Exception:
                pass

        self._hooks = TorchTrainingHooks(self, enable_debug_print=enable_debug_print)
        self._hooks.install(model=model, loss_fn=loss_fn, optimizer=optimizer)

    # ------------------------------------------------------------
    # Final Summary 
    # ------------------------------------------------------------

    def build_summary_dict(self) -> Dict[str, int]:
        return {
            "total_model_flop": self.total_model_flop,
            "total_optimizer_flop": self.total_optimizer_flop,
            "total_loss_forward_flop": self.total_loss_forward_flop,
            "total_loss_backward_flop": self.total_loss_backward_flop,
            "total_overall_flop": self.total_overall_flop,
        }
