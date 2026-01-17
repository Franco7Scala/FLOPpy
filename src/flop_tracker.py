from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


@dataclass
class FlopReport:
    run_name: Optional[str]
    backend: str

    model_flop: int

    # breakdown esplicito
    loss_flop: int
    preproc_ops: int

    # logging / integrazioni
    export_path: Optional[str]
    use_wandb: bool
    wandb_project: Optional[str]

    extra: Dict[str, Any]


class FlopTracker:
    """
    - Nasconde complessità di Tracker + backend + logger + training algorithm
    - Espone un'unica API di alto livello: run(...)
    - Il training rimane esterno e viene passato come funzione (Strategy-like)
    """

    def __init__(self, run_name: Optional[str] = None, *, print_summary: bool = True, print_hardware: bool = False):
        self.run_name = run_name
        self.print_summary = print_summary
        self.print_hardware = print_hardware
        self._report: Optional[FlopReport] = None

    @property
    def report(self) -> FlopReport:
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
        - train_fn: funzione di training esterna (es. trainers.train_torch / train_hf / train_sklearn)
        - train_kwargs: parametri specifici dell'algoritmo (optimizer, loss_fn, loader, ecc.)
        - train_fn deve accettare observers=[...]
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

            # Training esterno 
            train_fn(**train_kwargs, observers=[tr])

            model_flop = int(tr.total_flop)
            loss_flop = int(getattr(tr, "total_loss_flop", 0))
            preproc_ops = int(getattr(tr, "total_preproc_ops", 0))

            self._report = FlopReport(
                run_name=self.run_name,
                backend=backend,
                model_flop=model_flop,
                loss_flop=loss_flop,
                preproc_ops=preproc_ops,
                export_path=export_path,
                use_wandb=use_wandb,
                wandb_project=wandb_project,
                extra={**extra_ctx, "hardware": hw},
            )

        if self.print_summary:
            self._print_summary()

        return self

    def _print_summary(self) -> None:
        rep = self.report
        run_label = f"[{rep.run_name}]" if rep.run_name else ""

        if rep.extra.get("hardware") is not None:
            print(f"[FlopTracker{run_label}] Hardware: {rep.extra['hardware']}")

        # Questo è il totale FLOP del modello 
        print(f"[FlopTracker{run_label}] FLOP modello (incl. extra): {rep.model_flop}")

        # Breakdown esplicito
        if rep.loss_flop > 0:
            print(f"[FlopTracker{run_label}] FLOP loss (forward): {rep.loss_flop}")
        if rep.preproc_ops > 0:
            print(f"[FlopTracker{run_label}] Ops preprocessing/tokenizer: {rep.preproc_ops}")

        if rep.export_path:
            print(f"[FlopTracker{run_label}] Export CSV: {rep.export_path}")
        if rep.use_wandb and rep.wandb_project:
            print(f"[FlopTracker{run_label}] W&B project: {rep.wandb_project}")
