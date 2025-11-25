from __future__ import annotations
from contextlib import AbstractContextManager

from backends import create_backend
from ft_logging import create_logger


class Tracker(AbstractContextManager):
    """
    Tracker interno (non esposto direttamente all'utente finale).

    - Instanzia backend e logger
    - Aggancia gli hook
    - Espone total_flop
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

    def __enter__(self):
        self.backend.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.backend.stop()
        if self.logger is not None:
            self.logger.close()
        return False 
    @property
    def total_flop(self) -> int:
        return self.backend.get_total_flop()
