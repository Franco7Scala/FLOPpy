from __future__ import annotations
from contextlib import AbstractContextManager

from backends import create_backend
from ft_logging import create_logger


class Tracker(AbstractContextManager):
    """
    Tracker interno (non esposto direttamente all'utente finale).

    - Instanzia backend e logger
    - Aggancia gli hook
    - Espone total_flop (FLOPs del modello)
    - Tiene traccia opzionale delle operazioni di preprocessing/tokenizer
    """

    def __init__(
        self,
        model,
        backend: str = "auto",
        log_per_batch: bool = False,
        log_per_epoch: bool = False,
        export_path: str | None = None,
        use_wandb: bool = False,
        wandb_project: str | None = None,
        wandb_token: str | None = None,
        run_name: str | None = None,
    ):
        self.logger = create_logger(
            log_per_batch=log_per_batch,
            log_per_epoch=log_per_epoch,
            export_path=export_path,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_token=wandb_token,
            run_name=run_name,
        )

        self.backend = create_backend(model, backend, logger=self.logger)

        # --- contatori per operazioni di preprocessing/tokenizer 
        # registrati tramite add_preproc_ops(...).
        self.preproc_ops: int = 0
        self.preproc_ops_cumulative: int = 0

    # Context manager
    def __enter__(self):
        self.backend.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.backend.stop()
        if self.logger is not None:
            self.logger.close()
        return False

    # ---------------- FLOPs del modello  ----------------
    @property
    def total_flop(self) -> int:
        """
        Restituisce i FLOPs totali del MODELLO (valore fornito dal backend).
        Non include le operazioni di preprocessing/tokenizer.
        """
        return self.backend.get_total_flop()

    # -------------  API per le operazioni di preprocessing ----------

    def add_preproc_ops(self, ops: int) -> None:
        """
        Registra un numero di operazioni di preprocessing/tokenizer.

        Esempio:
            n_chars = ...
            n_tokens = ...
            tracker.add_preproc_ops(n_chars + n_tokens)
        """
        if ops is None or ops <= 0:
            return
        self.preproc_ops += int(ops)
        self.preproc_ops_cumulative += int(ops)

    @property
    def total_preproc_ops(self) -> int:
        """
        Restituisce il totale delle operazioni di preprocessing/tokenizer registrate.
        """
        return self.preproc_ops_cumulative

    @property
    def total_operations(self) -> float:
        """
        Restituisce un totale aggregato:
            FLOPs del modello + operazioni di preprocessing/tokenizer.
        """
        return float(self.total_flop + self.preproc_ops_cumulative)
