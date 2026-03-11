from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
from torch.optim import Optimizer
from backends.sklearn_backend import SklearnBackend


@dataclass
class FLOPpyReport:
    run_name: Optional[str]
    backend: str

    # breakdown esplicito
    model_flop: int
    optimizer_flop: int
    loss_forward_flop: int
    loss_backward_flop: int
    preproc_ops: int
    overall_flop: int

    # logging / integrazioni
    export_path: Optional[str]
    use_wandb: bool
    wandb_project: Optional[str]

    hardware: Optional[Dict[str, Any]]


class FLOPpyTracker:
    """
    Facade hook-only.

    Uso previsto:
        with FLOPpyTracker(...).run(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            export_path="flop.csv"
        ) as ft:
            ... training loop ...

    Il report viene stampato automaticamente a fine monitoraggio.
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
        self._tracker = None

        self._model = None
        self._optimizer: Optional[Optimizer] = None
        self._loss_fn: Optional[Any] = None

        self._export_path: Optional[str] = None
        self._use_wandb: bool = False
        self._wandb_project: Optional[str] = None
        self._wandb_token: Optional[str] = None
        self._hooks_debug_print: bool = False

        self._hardware: Optional[Dict[str, Any]] = None
        self._backend_name: str = "auto"

    @property
    def report(self) -> FLOPpyReport:
        if self._report is None:
            raise RuntimeError(
                "Nessun report disponibile: esegui il training dentro il context manager prima di accedere a report."
            )
        return self._report

    def run(
        self,
        *,
        model,
        optimizer: Optional[Optimizer] = None,
        loss_fn: Optional[Any] = None,
        export_path: Optional[str] = None,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_token: Optional[str] = None,
        hooks_debug_print: bool = False,
    ) -> FLOPpyTracker:
        """
        Prepara il monitoraggio e restituisce self come context manager.

        Esempio:
            with FLOPpyTracker(...).run(model=model, optimizer=opt, loss_fn=loss_fn) as ft:
                ... training loop ...
        """
        self._model = model
        self._optimizer = optimizer
        self._loss_fn = loss_fn
        self._export_path = export_path
        self._use_wandb = use_wandb
        self._wandb_project = wandb_project
        self._wandb_token = wandb_token
        self._hooks_debug_print = hooks_debug_print

        return self

    def __enter__(self) -> FLOPpyTracker:
        from tracker import Tracker

        if self._model is None:
            raise RuntimeError("run(...) deve essere chiamato prima di entrare nel context manager.")

        # hardware info opzionale
        if self.print_hardware:
            try:
                from hardware_info import get_hardware_info
                self._hardware = get_hardware_info()
            except Exception:
                self._hardware = None
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

        # sklearn non usa hook torch/loss/optimizer
        if not isinstance(self._tracker.backend, SklearnBackend):
            self._tracker.attach_torch_hooks(
                model=self._model,
                loss_fn=self._loss_fn,
                optimizer=self._optimizer,
                enable_debug_print=self._hooks_debug_print,
            )

        # nome backend reale
        self._backend_name = self._tracker.backend.__class__.__name__.replace("Backend", "").lower()

        return self

    def __exit__(self, exc_type, exc, tb):
        if self._tracker is not None:
            self._tracker.__exit__(exc_type, exc, tb)

            model_flop = int(getattr(self._tracker, "total_model_flop", 0))
            optimizer_flop = int(getattr(self._tracker, "total_optimizer_flop", 0))
            loss_forward_flop = int(getattr(self._tracker, "total_loss_forward_flop", 0))
            loss_backward_flop = int(getattr(self._tracker, "total_loss_backward_flop", 0))
            preproc_ops = int(getattr(self._tracker, "total_preproc_ops", 0))
            overall_flop = int(getattr(self._tracker, "total_overall_flop", 0))

            self._report = FLOPpyReport(
                run_name=self.run_name,
                backend=self._backend_name,
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

        if self.print_summary and self._report is not None:
            self._print_summary()

        return False

    def _print_summary(self) -> None:
        rep = self.report
        run_label = f"[{rep.run_name}]" if rep.run_name else ""

        if rep.hardware is not None:
            print(f"[FLOPpyTracker{run_label}] Hardware: {rep.hardware}")

        print(f"[FLOPpyTracker{run_label}] FLOP modello: {rep.model_flop}")

        if rep.loss_forward_flop > 0:
            print(f"[FLOPpyTracker{run_label}] FLOP loss forward: {rep.loss_forward_flop}")

        if rep.loss_backward_flop > 0:
            print(f"[FLOPpyTracker{run_label}] FLOP loss backward: {rep.loss_backward_flop}")

        if rep.optimizer_flop > 0:
            print(f"[FLOPpyTracker{run_label}] FLOP optimizer: {rep.optimizer_flop}")

        if rep.preproc_ops > 0:
            print(f"[FLOPpyTracker{run_label}] Ops preprocessing/tokenizer: {rep.preproc_ops}")

        print(f"[FLOPpyTracker{run_label}] FLOP totale complessivo: {rep.overall_flop}")

        if rep.export_path:
            print(f"[FLOPpyTracker{run_label}] Export CSV: {rep.export_path}")

        if rep.use_wandb and rep.wandb_project:
            print(f"[FLOPpyTracker{run_label}] W&B project: {rep.wandb_project}")
