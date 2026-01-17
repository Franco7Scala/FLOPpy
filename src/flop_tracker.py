from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


@dataclass
class FlopsReport:
    run_name: Optional[str]
    backend: str
    model_flops: int
    loss_flops: int
    preproc_ops: int
    total_operations: float
    export_path: Optional[str]
    use_wandb: bool
    wandb_project: Optional[str]
    extra: Dict[str, Any]


class FlopTracker:
    """
    sfrutto il Design Pattern Facade
    - Nasconde la complessità di Tracker + backend + logger + training algorithm
    - Espone un'unica API di alto livello: run(...)
    - Il training rimane esterno e viene passato come funzione (Strategy-like)
    """

    def __init__(self, run_name: Optional[str] = None, *, print_summary: bool = True, print_hardware: bool = False):
        self.run_name = run_name
        self.print_summary = print_summary
        self.print_hardware = print_hardware

        self._report: Optional[FlopsReport] = None

    @property
    def report(self) -> FlopsReport:
        if self._report is None:
            raise RuntimeError("Nessun report disponibile: esegui prima FlopTracker.run(...).")
        return self._report

    def run(
        self,
        *,
        model,
        train_fn: Callable[..., None],
        train_kwargs: Dict[str, Any],
        backend: str = "torch",
        # logging
        log_per_batch: bool = False,
        log_per_epoch: bool = False,
        export_path: Optional[str] = None,
        # wandb
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_token: Optional[str] = None,
        # extra
        extra_ctx: Optional[Dict[str, Any]] = None,
    ) -> "FlopTracker":
        """
        Esegue una run osservata dal Tracker.

        Parametri chiave:
        - train_fn: funzione di training esterna (es. trainers.train_torch)
        - train_kwargs: parametri specifici dell'algoritmo di training (optimizer, loss_fn, loader, ecc.)

        Il Tracker osserva tramite callbacks.
        """
        from tracker import Tracker  # import locale per evitare problemi di packaging

        extra_ctx = extra_ctx or {}

        # (opzionale) hardware info stile codecarbon
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
            log_per_batch=log_per_batch,
            log_per_epoch=log_per_epoch,
            export_path=export_path,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_token=wandb_token,
            run_name=self.run_name,
        ) as tr:

            # Esecuzione del training esterno (il Tracker è un observer)
            # Convenzione: train_fn deve accettare observers=[...]
            train_fn(**train_kwargs, observers=[tr])

            # costruzione report
            self._report = FlopsReport(
                run_name=self.run_name,
                backend=backend,
                model_flops=int(tr.total_flop),
                loss_flops=int(getattr(tr, "total_loss_flop", 0)),
                preproc_ops=int(getattr(tr, "total_preproc_ops", 0)),
                total_operations=float(getattr(tr, "total_operations", tr.total_flop)),
                export_path=export_path,
                use_wandb=use_wandb,
                wandb_project=wandb_project,
                extra={
                    **extra_ctx,
                    "hardware": hw,
                },
            )

        if self.print_summary:
            self._print_summary()

        return self

    def _print_summary(self) -> None:
        rep = self.report
        run_label = f"[{rep.run_name}]" if rep.run_name else ""

        if rep.extra.get("hardware") is not None:
            print(f"[FlopTracker{run_label}] Hardware: {rep.extra['hardware']}")

        print(f"[FlopTracker{run_label}] FLOPs modello: {rep.model_flops}")
        if rep.loss_flops > 0:
            print(f"[FlopTracker{run_label}] FLOPs loss (forward): {rep.loss_flops}")
        if rep.preproc_ops > 0:
            print(f"[FlopTracker{run_label}] Ops preprocessing/tokenizer: {rep.preproc_ops}")

        print(f"[FlopTracker{run_label}] Totale operazioni (model + preproc): {rep.total_operations:.0f}")
        if rep.export_path:
            print(f"[FlopTracker{run_label}] Export CSV: {rep.export_path}")
        if rep.use_wandb and rep.wandb_project:
            print(f"[FlopTracker{run_label}] W&B project: {rep.wandb_project}")
