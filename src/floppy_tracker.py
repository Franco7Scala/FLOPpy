from __future__ import annotations
from torch.nn import nn
from dataclasses import dataclass
from typing import Any, Dict, Optional
from backends.sklearn_backend import SklearnBackend


@dataclass
class FLOPpyReport:
    run_name: Optional[str]
    backend: str

    # totale backend (modello + extra aggiunti via add_extra_flop, es. loss)
    model_flop: int

    # breakdown esplicito (solo per stampa/report)
    optimizer_flop: int
    loss_flop: int
    preproc_ops: int

    # logging / integrazioni
    export_path: Optional[str]
    use_wandb: bool
    wandb_project: Optional[str]

    hardware: Dict[str, Any]


class FLOPpyTracker:
    """
    Facade:
    - Nasconde complessità di Tracker + backend + logger + training algorithm
    - Espone un'unica API di alto livello: run(...)
    - Supporta due modalità user-friendly:
        A) Observer (train_fn accetta observers=[...])  -> default
        B) Hook-based (train_fn NON serve observer-aware) -> attivi via use_training_hooks=True (solo torch)
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
            raise RuntimeError("Nessun report disponibile: esegui prima FlopTracker.run(...).")
        return self._report

    def run(
        self,
        model,
        optimizer: Optional[nn.Module] = None,
        loss_fn: Optional[nn.Module] = None,
        export_path: Optional[str] = None,
        # wandb
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_token: Optional[str] = None,
    ) -> FLOPpyTracker:
        """
        Esegue una run osservata dal Tracker.

        Modalità:
        - use_training_hooks=False (default):
            train_fn DEVE accettare observers=[...]
        - use_training_hooks=True (solo torch):
            train_fn può essere "vanilla" (non observer-aware).
            Il Tracker installa hook su model/loss/optimizer e raccoglie i costi extra.
            In questo caso è consigliato passare in train_kwargs anche:
              - loss_fn
              - optimizer
        """
        from tracker import Tracker  # import locale per evitare problemi di packaging

        # (opzionale) hardware info stile codecarbon
        hw = None
        if self.print_hardware:
            from hardware_info import get_hardware_info
            hw = get_hardware_info()

        with Tracker(
            model=model,
            export_path=export_path,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_token=wandb_token,
            run_name=self.run_name,
        ) as tr:

            # ------------------ modalità HOOKS (torch) ------------------ #
            if not tr.backend is SklearnBackend:
                tr.attach_torch_hooks(
                    model=model,
                    loss_fn=loss_fn,
                    optimizer=optimizer
                )

            # report
            model_flop = int(tr.total_flop)
            optimizer_flop = int(getattr(tr, "total_optimizer_flop", 0))
            loss_flop = int(getattr(tr, "total_loss_flop", 0))
            preproc_ops = int(getattr(tr, "total_preproc_ops", 0))

            self._report = FLOPpyReport(
                run_name=self.run_name,
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

        if rep.extra.get("hardware") is not None:
            print(f"[FlopTracker{run_label}] Hardware: {rep.extra['hardware']}")

        # totale backend (modello + extra)
        print(f"[FlopTracker{run_label}] FLOP modello (incl. extra): {rep.model_flop}")

        # breakdown esplicito
        if rep.loss_flop > 0:
            print(f"[FlopTracker{run_label}] FLOP loss (forward): {rep.loss_flop}")
        if rep.preproc_ops > 0:
            print(f"[FlopTracker{run_label}] Ops preprocessing/tokenizer: {rep.preproc_ops}")

        if rep.export_path:
            print(f"[FlopTracker{run_label}] Export CSV: {rep.export_path}")
        if rep.use_wandb and rep.wandb_project:
            print(f"[FlopTracker{run_label}] W&B project: {rep.wandb_project}")
