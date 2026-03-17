from __future__ import annotations
from .csv_logger import CsvLogger
from .wandb_logger import WandbLogger


class CompositeLogger:
    def __init__(self, loggers):
        self.loggers = [l for l in loggers if l is not None]

    def log_summary(self, summary: dict):
        result = summary
        for logger in self.loggers:
            try:
                result = logger.log_summary(summary)
            except Exception:
                pass

        return result

    def close(self):
        for logger in self.loggers:
            try:
                logger.close()
            except Exception:
                pass


def create_logger(
    export_path: str | None = None,
    use_wandb: bool = False,
    wandb_project: str | None = None,
    wandb_token: str | None = None,
    run_name: str | None = None,
):
    csv_logger = CsvLogger(export_path=export_path) if export_path is not None else None
    wandb_logger = (
        WandbLogger(
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_token=wandb_token,
            run_name=run_name,
        )
        if use_wandb
        else None
    )

    active = [l for l in (csv_logger, wandb_logger) if l is not None]

    if not active:
        return None

    if len(active) == 1:
        return active[0]

    return CompositeLogger(active)
