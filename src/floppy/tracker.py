from __future__ import annotations
from typing import Any, Optional
from torch.optim import Optimizer
from .core import Tracker
from .backends.sklearn_backend import SklearnBackend
from .utils.floppy_report import FLOPpyReport
from .utils.hardware_info import get_hardware_info, HardwareInfo
from .utils.tokenizer_ops import TokenizerWithOps
from .utils.wandb_configuration import WandbConfiguration


class FLOPpyTracker:
    """
    Tracker for monitoring FLOPs in machine learning and deep learning models.
    for models, optimizers, loss functions and tokenizers during training or inference.
    It supports integration with Weights & Biases for logging and can export reports.

    Supported usage patterns:
    1) Immediate-start mode:
        tracker = FLOPpyTracker(...)
        tracker.run(model=model, optimizer=optimizer, loss_fn=loss_fn)
        ...
        tracker.batch()
        tracker.epoch()
        print(tracker.report())

    2) Context-manager mode:
        with FLOPpyTracker(...) as tracker:
            tracker.start(model=model, optimizer=optimizer, loss_fn=loss_fn)
            ...
            tracker.batch()
            tracker.epoch()
            tracker.stop()

    Methods:
        run(...): Configure and immediately start monitoring.
        start(...): Start monitoring.
        stop(): Stop monitoring and generate the report.
        report(): Get the final report.
        batch(): Log a batch snapshot.
        epoch(): Log an epoch snapshot.
    """

    def __init__(self, run_name: Optional[str] = None):
        self.run_name = run_name
        self._report: Optional[FLOPpyReport] = None
        self._tracker: Optional[Tracker] = None
        self._model = None
        self._optimizer: Optional[Optimizer] = None
        self._loss_fn: Optional[Any] = None
        self._base_tokenizer: Optional[Any] = None
        self._wrapped_tokenizer: Optional[TokenizerWithOps] = None
        self._export_path: Optional[str] = None
        self._wandb_config: Optional[WandbConfiguration] = None
        self._hooks_debug_print: bool = False
        self._hardware: Optional[HardwareInfo] = None
        self._is_active: bool = False
        self._summary_printed: bool = False

        # progress counters for callback-based logging
        self._epoch_idx: int = 0
        self._batch_idx: int = 0

    # ------------------------------------------------------------
    # Tokenizer access
    # ------------------------------------------------------------

    @property
    def tokenizer(self) -> TokenizerWithOps:
        """
        Returns the wrapped tokenizer connected to the internal Tracker.
        Available only after start/run if a tokenizer was provided.
        """
        if self._wrapped_tokenizer is None:
            raise RuntimeError("No wrapped tokenizer available. Pass tokenizer=... to start(...) or run(...).")
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
        wandb_config: Optional[WandbConfiguration] = None
    ) -> FLOPpyTracker:
        """
        Configure and immediately start monitoring.

        Args:
            model: The model to monitor.
            optimizer (Optional[Optimizer]): The optimizer to monitor.
            loss_fn (Optional[Any]): The loss function to monitor.
            tokenizer (Optional[Any]): The tokenizer to monitor.
            export_path (Optional[str]): Local path to export the report.
            wandb_config (Optional[WandbConfiguration]): Weights & Biases configuration containing project name and token.

        Returns:
            FLOPpyTracker: The tracker instance.
        """
        return self.start(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            tokenizer=tokenizer,
            export_path=export_path,
            wandb_config=wandb_config
        )

    def start(
        self,
        model=None,
        optimizer: Optional[Optimizer] = None,
        loss_fn: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        export_path: Optional[str] = None,
        wandb_config: Optional[WandbConfiguration] = None
    ) -> FLOPpyTracker:
        """
        Start monitoring.
        It can be used directly inside a context manager:
            with FLOPpyTracker(...) as tracker:
                tracker.start(model=..., optimizer=..., loss_fn=..., tokenizer=...)
                ...
                tracker.stop()
        """
        if self._is_active:
            return self

        if model is not None:
            self._model = model

        self._optimizer = optimizer
        self._loss_fn = loss_fn
        self._base_tokenizer = tokenizer
        self._export_path = export_path
        self._wandb_config = wandb_config
        self._hooks_debug_print = False

        if self._model is None:
            raise RuntimeError("A model must be provided before starting monitoring.")

        self._report = None
        self._summary_printed = False
        self._wrapped_tokenizer = None

        # Reset logging counters at every new run
        self._epoch_idx = 0
        self._batch_idx = 0
        self._hardware = get_hardware_info()
        self._tracker = Tracker(
            model=self._model,
            backend="auto",
            export_path=self._export_path,
            wandb_config=self._wandb_config,
            run_name=self.run_name,
        )
        self._tracker.__enter__()

        # Automatic tokenizer wrapping for HF / preprocessing
        if self._base_tokenizer is not None:
            self._wrapped_tokenizer = self._tracker.wrap_tokenizer(self._base_tokenizer)

        # Torch-specific hooks
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
    # Intermediate logging callbacks
    # ------------------------------------------------------------

    def _build_progress_snapshot(self) -> dict:
        """
        Build the current intermediate logging snapshot.
        """
        if self._tracker is None:
            raise RuntimeError("No active internal tracker available.")

        snapshot = self._tracker.build_progress_dict()
        snapshot["epoch_idx"] = self._epoch_idx
        snapshot["batch_idx"] = self._batch_idx
        return snapshot

    def batch(self) -> dict:
        """
        Log and returns the current FLOP counters as a batch snapshot.
        Each call increments the internal batch counter by 1.
        """
        if not self._is_active or self._tracker is None:
            raise RuntimeError("batch() requires an active monitoring session.")

        snapshot = self._build_progress_snapshot()
        if self._tracker.logger is not None and hasattr(self._tracker.logger, "log_batch"):
            self._tracker.logger.log_batch(snapshot)

        self._batch_idx += 1
        return snapshot

    def epoch(self) -> dict:
        """
        Log and returns the current FLOP counters as an epoch snapshot.
        Each call increments the internal epoch counter by 1.
        """
        if not self._is_active or self._tracker is None:
            raise RuntimeError("epoch() requires an active monitoring session.")

        snapshot = self._build_progress_snapshot()
        if self._tracker.logger is not None and hasattr(self._tracker.logger, "log_epoch"):
            self._tracker.logger.log_epoch(snapshot)

        self._epoch_idx += 1
        return snapshot

    # ------------------------------------------------------------
    # Stop + report
    # ------------------------------------------------------------

    def stop(self, print_summary=False) -> FLOPpyTracker:
        """
        Stop monitoring.
        Safe to call multiple times.
        """
        if not self._is_active:
            return self

        internal_tracker = self._tracker
        if internal_tracker is not None:
            internal_tracker.__exit__(None, None, None)

        self._build_report(internal_tracker)
        if print_summary and self._report is not None and not self._summary_printed:
            print(self._report)
            self._summary_printed = True

        self._tracker = None
        self._is_active = False
        return self

    def report(self) -> FLOPpyReport:
        """
        Returns a report.
        """
        if not self._is_active and self._report is not None:
            return self._report

        elif self._is_active:
            self._build_report(self._tracker)

        if self._report is None:
            raise RuntimeError("No report available. Start monitoring before requesting a report.")

        return self._report

    def _build_report(self, tracker_obj: Optional[Tracker] = None) -> None:
        """
        Internal helper to build the final FLOPpyReport.
        """
        if tracker_obj is None:
            tracker_obj = self._tracker

        if tracker_obj is None:
            return

        model_flop = int(getattr(tracker_obj, "total_model_flop", 0))
        optimizer_flop = int(getattr(tracker_obj, "total_optimizer_flop", 0))
        loss_forward_flop = int(getattr(tracker_obj, "total_loss_forward_flop", 0))
        loss_backward_flop = int(getattr(tracker_obj, "total_loss_backward_flop", 0))
        preproc_ops = int(getattr(tracker_obj, "total_preproc_ops", 0))
        overall_flop = int(getattr(tracker_obj, "total_overall_flop", 0))
        model_architecture = "unknown"
        model_device = "CPU"
        if self._model is not None:
            model_cls = self._model.__class__
            model_architecture = f"{model_cls.__module__}.{model_cls.__name__}"

            if hasattr(self._model, "parameters"):
                try:
                    param_device = next(self._model.parameters()).device
                    model_device = str(param_device).upper()

                except Exception:
                    model_device = "Unknown"

        self._report = FLOPpyReport(
            run_name=self.run_name,
            backend=tracker_obj.backend.__class__.__name__.replace("Backend", "").lower(),
            model_architecture=model_architecture,
            loss_type=f"{getattr(self._loss_fn, '__name__', self._loss_fn.__class__.__name__)}" if self._loss_fn is not None else None,
            optimizer_type=f"{getattr(self._optimizer, '__name__', self._optimizer.__class__.__name__)}" if self._optimizer is not None else None,
            model_device=model_device,
            model_flop=model_flop,
            optimizer_flop=optimizer_flop,
            loss_forward_flop=loss_forward_flop,
            loss_backward_flop=loss_backward_flop,
            preproc_ops=preproc_ops,
            overall_flop=overall_flop,
            export_path=self._export_path,
            wandb_config=self._wandb_config,
            hardware=self._hardware
        )
   
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

    def wrap_tokenizer(self, base_tokenizer, cost_model: str = "chars+tokens") -> TokenizerWithOps:
        """
        Optional helper for manual tokenizer wrapping.
        Requires an active internal Tracker.
        """
        if self._tracker is None:
            raise RuntimeError(
                "wrap_tokenizer(...) requires an active Tracker."
                "Use start(...) or run(...) first."
            )

        return self._tracker.wrap_tokenizer(
            base_tokenizer=base_tokenizer,
            cost_model=cost_model,
        )
