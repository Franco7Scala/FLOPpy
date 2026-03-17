from __future__ import annotations
from typing import Any, Dict, Optional
from torch.optim import Optimizer
from core import Tracker
from backends.sklearn_backend import SklearnBackend
from utils.floppy_report import FLOPpyReport
from utils.hardware_info import get_hardware_info
from utils.tokenizer_ops import TokenizerWithOps


class FLOPpyTracker:
    def __init__(
        self,
        run_name: Optional[str] = None,
        print_summary: bool = True,
        print_hardware: bool = False,
    ):
        self.run_name = run_name
        self.print_summary = print_summary
        self.print_hardware = print_hardware
        self._report: Optional[FLOPpyReport] = None
        self._tracker: Optional[Tracker] = None
        self._model = None
        self._optimizer: Optional[Optimizer] = None
        self._loss_fn: Optional[Any] = None
        self._base_tokenizer: Optional[Any] = None
        self._wrapped_tokenizer: Optional[TokenizerWithOps] = None
        self._export_path: Optional[str] = None
        self._use_wandb: bool = False
        self._wandb_project: Optional[str] = None
        self._wandb_token: Optional[str] = None
        self._hooks_debug_print: bool = False
        self._hardware: Optional[Dict[str, Any]] = None
        self._is_active: bool = False
        self._summary_printed: bool = False

    @property
    def tokenizer(self) -> TokenizerWithOps:
        """
        Returns the wrapped tokenizer connected to the internal Tracker.
        Available only after start/run if a tokenizer was provided.
        """
        if self._wrapped_tokenizer is None:
            raise RuntimeError(
                "No tokenizer available. Pass tokenizer=... to start(...) or run(...)."
            )
        return self._wrapped_tokenizer

    # ------------------------------------------------------------
    # Configuration + start
    # ------------------------------------------------------------

    def run(
        self,
        model,
        optimizer: Optional[Optimizer] = None,
        loss_fn: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        export_path: Optional[str] = None,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_token: Optional[str] = None,
        hooks_debug_print: bool = False,
    ) -> FLOPpyTracker:
        """
        Configure and immediately start monitoring.
        """
        return self.start(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            tokenizer=tokenizer,
            export_path=export_path,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_token=wandb_token,
            hooks_debug_print=hooks_debug_print,
        )

    def start(
        self,
        model=None,
        optimizer: Optional[Optimizer] = None,
        loss_fn: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        export_path: Optional[str] = None,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_token: Optional[str] = None,
        hooks_debug_print: bool = False,
    ) -> FLOPpyTracker:
        """
        Start monitoring.

        It can be used directly inside a context manager:
            with FLOPpyTracker(...) as tracker:
                tracker.start(model=..., optimizer=..., loss_fn=...)
                ...
                tracker.stop()
        """
        if self._is_active:
            return self

        if model is not None:
            self._model = model
        if optimizer is not None:
            self._optimizer = optimizer
        if loss_fn is not None:
            self._loss_fn = loss_fn
        if tokenizer is not None:
            self._base_tokenizer = tokenizer

        if export_path is not None:
            self._export_path = export_path

        self._use_wandb = use_wandb
        self._wandb_project = wandb_project
        self._wandb_token = wandb_token
        self._hooks_debug_print = hooks_debug_print

        if self._model is None:
            raise RuntimeError("A model must be provided before starting monitoring.")

        self._report = None
        self._summary_printed = False
        self._wrapped_tokenizer = None

        if self.print_hardware:
            self._hardware = get_hardware_info()
        else:
            self._hardware = None

        self._tracker = Tracker(
            model=self._model,
            backend="auto",
            export_path=self._export_path,
            use_wandb=self._use_wandb,
            wandb_project=self._wandb_project,
            wandb_token=self._wandb_token,
            run_name=self.run_name,
        )
        self._tracker.__enter__()

        if self._base_tokenizer is not None:
            self._wrapped_tokenizer = self._tracker.wrap_tokenizer(self._base_tokenizer)

        if not isinstance(self._tracker.backend, SklearnBackend):
            self._tracker.attach_torch_hooks(
                model=self._model,
                loss_fn=self._loss_fn,
                optimizer=self._optimizer,
                enable_debug_print=self._hooks_debug_print,
            )

        self._is_active = True
        return self

    # ------------------------------------------------------------
    # Stop + report
    # ------------------------------------------------------------

    def stop(self) -> FLOPpyTracker:
        """
        Stop monitoring and build the final report.
        Safe to call multiple times.
        """
        if not self._is_active:
            return self

        if self._tracker is not None:
            self._tracker.__exit__(None, None, None)

            model_flop = int(getattr(self._tracker, "total_model_flop", 0))
            optimizer_flop = int(getattr(self._tracker, "total_optimizer_flop", 0))
            loss_forward_flop = int(getattr(self._tracker, "total_loss_forward_flop", 0))
            loss_backward_flop = int(getattr(self._tracker, "total_loss_backward_flop", 0))
            preproc_ops = int(getattr(self._tracker, "total_preproc_ops", 0))
            overall_flop = int(getattr(self._tracker, "total_overall_flop", 0))

            self._report = FLOPpyReport(
                run_name=self.run_name,
                backend=self._tracker.backend.__class__.__name__.replace("Backend", "").lower(),
                model_flop=model_flop,
                optimizer_flop=optimizer_flop,
                loss_forward_flop=loss_forward_flop,
                loss_backward_flop=loss_backward_flop,
                preproc_ops=preproc_ops,
                overall_flop=overall_flop,
                export_path=self._export_path,
                use_wandb=self._use_wandb,
                wandb_project=self._wandb_project,
                hardware=self._hardware,
            )

        self._tracker = None
        self._is_active = False

        if self.print_summary and self._report is not None and not self._summary_printed:
            self._print_summary()
            self._summary_printed = True

        return self

    def report(self) -> FLOPpyReport:
        """
        Return the final report.
        If monitoring is still active, stop it first.
        """
        if self._is_active:
            self.stop()

        if self._report is None:
            raise RuntimeError("No report available: start/run monitoring before requesting the report.")

        return self._report

    # ------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------

    def __enter__(self) -> FLOPpyTracker:
        """
        Returns the facade itself.
        Monitoring is started explicitly via start(...) or run(...).
        """
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False

    # ------------------------------------------------------------
    # Manual tokenizer wrapping (optional helper)
    # ------------------------------------------------------------

    def wrap_tokenizer(
        self,
        base_tokenizer,
        cost_model: str = "chars+tokens",
    ) -> TokenizerWithOps:
        """
        Optional helper for manual tokenizer wrapping.
        Requires an active internal Tracker.
        """
        if self._tracker is None:
            raise RuntimeError(
                "wrap_tokenizer(...) requires an active Tracker. "
                "Use start(...) or run(...) first."
            )

        return self._tracker.wrap_tokenizer(
            base_tokenizer=base_tokenizer,
            cost_model=cost_model,
        )

    # ------------------------------------------------------------
    # Printing
    # ------------------------------------------------------------

    def _print_summary(self) -> None:
        rep = self._report
        if rep is None:
            return

        run_label = f"[{rep.run_name}]" if rep.run_name else ""

        if rep.hardware is not None:
            print(f"[FLOPpyTracker{run_label}] Hardware: {rep.hardware}")

        print(f"[FLOPpyTracker{run_label}] model FLOPs: {rep.model_flop}")

        if rep.loss_forward_flop > 0:
            print(f"[FLOPpyTracker{run_label}] loss forward FLOPs: {rep.loss_forward_flop}")

        if rep.loss_backward_flop > 0:
            print(f"[FLOPpyTracker{run_label}] loss backward FLOPs: {rep.loss_backward_flop}")

        if rep.optimizer_flop > 0:
            print(f"[FLOPpyTracker{run_label}] optimizer FLOPs: {rep.optimizer_flop}")

        if rep.preproc_ops > 0:
            print(f"[FLOPpyTracker{run_label}] preprocessing/tokenizer Ops: {rep.preproc_ops}")

        print(f"[FLOPpyTracker{run_label}] overall FLOPs: {rep.overall_flop}")

        if rep.export_path:
            print(f"[FLOPpyTracker{run_label}] Export CSV: {rep.export_path}")

        if rep.use_wandb and rep.wandb_project:
            print(f"[FLOPpyTracker{run_label}] W&B project: {rep.wandb_project}")
