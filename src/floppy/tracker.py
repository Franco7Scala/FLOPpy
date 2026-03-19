from __future__ import annotations
from typing import Any, Optional
from torch.optim import Optimizer
from .core import Tracker
from .backends.sklearn_backend import SklearnBackend
from .utils.floppy_report import FLOPpyReport
from .utils.hardware_info import get_hardware_info, HardwareInfo
from .utils.tokenizer_ops import TokenizerWithOps


class FLOPpyTracker:
    """
    Tracker for monitoring FLOPs in machine learning and deep learning models.
    This class provides functionality to track floating-point operations (FLOPs)
    for models, optimizers, loss functions, and tokenizers during training or inference.

    Supported usage patterns:
    
    1) Immediate-start mode:
        tracker = FLOPpyTracker(...)
        tracker.run(model=model, optimizer=optimizer, loss_fn=loss_fn)
        ...
        print(tracker.report())
        
    2) Context-manager mode:
        with FLOPpyTracker(...) as tracker:
            tracker.start(model=model, optimizer=optimizer, loss_fn=loss_fn)
            ...
            tracker.stop()
    """

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
        self._hardware: Optional[HardwareInfo] = None
        self._is_active: bool = False
        self._summary_printed: bool = False

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
            raise RuntimeError(
                "No wrapped tokenizer available. Pass tokenizer=... to start(...) or run(...)."
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
                tracker.start(model=..., optimizer=..., loss_fn=..., tokenizer=...)
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

        # --------------------------------------------------------
        # Automatic tokenizer wrapping for HF / preprocessing
        # --------------------------------------------------------
        if self._base_tokenizer is not None:
            self._wrapped_tokenizer = self._tracker.wrap_tokenizer(self._base_tokenizer)

        # --------------------------------------------------------
        # Torch-specific hooks 
        # --------------------------------------------------------
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
        Stop monitoring.
        Safe to call multiple times.
        """
        if not self._is_active:
            return self

        if self._tracker is not None:
            self._tracker.__exit__(None, None, None)

        # Build report before clearing active state
        self._build_report()

        if self.print_summary and self._report is not None and not self._summary_printed:
            self._print_summary()
            self._summary_printed = True

        self._is_active = False
        return self

    def report(self) -> FLOPpyReport:
        """
        Returns the final report.
        If monitoring is still active, it is stopped automatically first.
        This supports the usage pattern:
            tracker.run(...)
            ...
            print(tracker.report())
        """
        if self._is_active:
            self.stop()

        if self._report is None:
            self._build_report()

        if self._report is None:
            raise RuntimeError("No report available. Start monitoring before requesting a report.")

        return self._report

    def _build_report(self) -> None:
        """
        Internal helper to build the final FLOPpyReport.
        """
        if self._tracker is None:
            return

        model_flop = int(getattr(self._tracker, "total_model_flop", 0))
        optimizer_flop = int(getattr(self._tracker, "total_optimizer_flop", 0))
        loss_forward_flop = int(getattr(self._tracker, "total_loss_forward_flop", 0))
        loss_backward_flop = int(getattr(self._tracker, "total_loss_backward_flop", 0))
        preproc_ops = int(getattr(self._tracker, "total_preproc_ops", 0))
        overall_flop = int(getattr(self._tracker, "total_overall_flop", 0))

        # Determine model architecture and device
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
            backend=self._tracker.backend.__class__.__name__.replace("Backend", "").lower(),
            model_architecture=model_architecture,
            model_device=model_device,
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

        def format_flops(flops: int) -> str:
            if flops == 0:
                return "0 FLOPs"

            units = ["FLOPs", "KFLOPs", "MFLOPs", "GFLOPs", "TFLOPs", "PFLOPs"]
            unit_idx = 0
            float_flops = float(flops)

            while float_flops >= 1000.0 and unit_idx < len(units) - 1:
                float_flops /= 1000.0
                unit_idx += 1

            return f"{float_flops:.2f} {units[unit_idx]}"

        run_label = f" '{rep.run_name}'" if rep.run_name else ""

        print("=" * 70)
        print(f" FLOPpyTracker Summary{run_label}")
        print("=" * 70)

        # Hardware info
        if rep.hardware is not None:
            h = rep.hardware
            print("Hardware Environment:")

            ram_str = f"{h.ram_total_gb:.0f} GB RAM" if h.ram_total_gb else "Unknown RAM"
            print(f"  - System   : {h.os} ({h.machine}) | {ram_str}")

            c_name = getattr(h, "cpu_name", None) or h.processor or "Unknown CPU"
            cores_str = (
                f"{h.cpu_cores_physical} Physical Cores"
                if h.cpu_cores_physical
                else "Unknown Cores"
            )
            print(f"  - CPU      : {c_name} | {cores_str}")

            if h.cuda_available:
                g_count = h.gpu_count or 1
                g_name = h.gpu_name or "Unknown GPU"
                print(f"  - GPU      : {g_count}x {g_name}")
            else:
                print("  - GPU      : None (CPU Only)")

            print(f"  - Python   : {h.python_version}")

            frameworks = []
            if getattr(h, "torch_version", None):
                frameworks.append(f"PyTorch {h.torch_version}")

            if getattr(h, "sklearn_version", None):
                frameworks.append(f"Scikit-learn {h.sklearn_version}")

            if frameworks:
                print(f"  - Libs     : {' | '.join(frameworks)}")

        # Model and device details
        print("Model details:")
        print(f"  - Model    : {rep.model_architecture}")
        print(f"  - Device   : {rep.model_device}")

        # Computational workload
        print("Computational Workload Breakdown:")
        print(f"  - Model (Forward)         : {format_flops(rep.model_flop):>15}")

        if rep.loss_forward_flop > 0:
            print(f"  - Loss (Forward)          : {format_flops(rep.loss_forward_flop):>15}")

        if rep.loss_backward_flop > 0:
            print(f"  - Loss (Backward)         : {format_flops(rep.loss_backward_flop):>15}")

        if rep.optimizer_flop > 0:
            print(f"  - Optimizer (Update)      : {format_flops(rep.optimizer_flop):>15}")

        if rep.preproc_ops > 0:
            print(f"  - Preprocessing/Tokenizer : {str(rep.preproc_ops) + ' Ops':>15}")

        print("-" * 70)
        print(f"OVERALL TOTAL FLOPs         : {format_flops(rep.overall_flop):>15}")
        print("=" * 70)

        if rep.export_path or (rep.use_wandb and rep.wandb_project):
            print("Tracking & Integrations:")

            if rep.export_path:
                print(f"  - Export Path: {rep.export_path}")

            if rep.use_wandb and rep.wandb_project:
                print(f"  - W&B Project: {rep.wandb_project}")

            print("=" * 70)
