from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from torch.optim import Optimizer

from backends.sklearn_backend import SklearnBackend


@dataclass
class FLOPpyReport:
    run_name: Optional[str]
    backend: str

    # totale backend (modello + extra aggiunti via add_extra_flop, es. loss)
    model_flop: int

    # breakdown esplicito
    optimizer_flop: int
    loss_flop: int
    preproc_ops: int

    # logging / integrazioni
    export_path: Optional[str]
    use_wandb: bool
    wandb_project: Optional[str]

    hardware: Optional[Dict[str, Any]]


class FLOPpyTracker:
    """
    Facade hook-only.

    Obiettivi:
    - nasconde la complessità di Tracker + backend + logger + training hooks
    - espone un'unica API di alto livello: run(...)
    - l'utente fornisce un train_fn 
    - il tracking avviene automaticamente via hook
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

    @property
    def report(self) -> FLOPpyReport:
        if self._report is None:
            raise RuntimeError("Nessun report disponibile: esegui prima FLOPpyTracker.run(...).")
        return self._report

    def run(
        self,
        *,
        model,
        train_fn: Optional[Callable[..., None]] = None,
        train_kwargs: Optional[Dict[str, Any]] = None,
        backend: str = "auto",
        optimizer: Optional[Optimizer] = None,
        loss_fn: Optional[Any] = None,
        export_path: Optional[str] = None,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_token: Optional[str] = None,
        hooks_debug_print: bool = False,
    ) -> FLOPpyTracker:
        """
        Esegue una run hook-based.

        Parametri:
        - model: modello da tracciare
        - train_fn: funzione di training
        - train_kwargs: parametri da passare a train_fn
        - backend: 'torch', 'sklearn', 'auto'
        - optimizer: ottimizzatore (serve per step hook)
        - loss_fn: loss module (serve per loss forward/backward hooks)
        """
        from tracker import Tracker

        train_kwargs = train_kwargs or {}

        # hardware info opzionale
        hw = None
        if self.print_hardware:
            try:
                from hardware_info import get_hardware_info
                hw = get_hardware_info()
            except Exception:
                hw = None

        with Tracker(
            model=model,
            backend=backend,
            export_path=export_path,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_token=wandb_token,
            run_name=self.run_name,
        ) as tr:

            # hook torch/hf: sklearn non li usa
            if not isinstance(tr.backend, SklearnBackend):
                tr.attach_torch_hooks(
                    model=model,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    enable_debug_print=hooks_debug_print,
                )

            # esegui training mentre gli hook sono attivi
            if train_fn is not None:
                train_fn(**train_kwargs)

            # report finale
            model_flop = int(tr.total_flop)
            # per ora optimizer FLOP non viene stimato: lo lasciamo a 0
            optimizer_flop = 0
            loss_flop = int(getattr(tr, "total_loss_flop", 0))
            preproc_ops = int(getattr(tr, "total_preproc_ops", 0))

            self._report = FLOPpyReport(
                run_name=self.run_name,
                backend=backend,
                model_flop=model_flop,
                optimizer_flop=optimizer_flop,
                loss_flop=loss_flop,
                preproc_ops=preproc_ops,
                export_path=export_path,
                use_wandb=use_wandb,
                wandb_project=wandb_project,
                hardware=hw,
            )

        if self.print_summary:
            self._print_summary()

        return self

    def _print_summary(self) -> None:
        rep = self.report
        run_label = f"[{rep.run_name}]" if rep.run_name else ""

        if rep.hardware is not None:
            print(f"[FLOPpyTracker{run_label}] Hardware: {rep.hardware}")

        print(f"[FLOPpyTracker{run_label}] FLOP modello (incl. extra): {rep.model_flop}")

        if rep.loss_flop > 0:
            print(f"[FLOPpyTracker{run_label}] FLOP loss (forward): {rep.loss_flop}")

        if rep.optimizer_flop > 0:
            print(f"[FLOPpyTracker{run_label}] FLOP optimizer: {rep.optimizer_flop}")

        if rep.preproc_ops > 0:
            print(f"[FLOPpyTracker{run_label}] Ops preprocessing/tokenizer: {rep.preproc_ops}")

        if rep.export_path:
            print(f"[FLOPpyTracker{run_label}] Export CSV: {rep.export_path}")

        if rep.use_wandb and rep.wandb_project:
            print(f"[FLOPpyTracker{run_label}] W&B project: {rep.wandb_project}")
