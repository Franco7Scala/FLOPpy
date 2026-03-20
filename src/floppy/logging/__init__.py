from __future__ import annotations
from .csv_logger import CsvLogger
from .wandb_logger import WandbLogger


class CompositeLogger:

    def __init__(self, loggers):
        self.loggers = [l for l in loggers if l is not None]

    def log_batch(self, summary: dict):
        result = summary
        for logger in self.loggers:
            result = logger.log_batch(summary)

        return result

    def log_epoch(self, summary: dict):
        result = summary
        for logger in self.loggers:
            result = logger.log_epoch(summary)

        return result

    def log_summary(self, summary: dict):
        result = summary
        for logger in self.loggers:
            result = logger.log_summary(summary)

        return result

    def close(self):
        for logger in self.loggers:
            logger.close()


def create_logger(
    export_path: str | None = None,
    use_wandb: bool = False,
    wandb_project: str | None = None,
    wandb_token: str | None = None,
    run_name: str | None = None,
):
    loggers = []
    if export_path is not None:
        loggers.append(CsvLogger(export_path=export_path))

    if use_wandb:
        loggers.append(WandbLogger(wandb_project=wandb_project, wandb_token=wandb_token, run_name=run_name))

    active = [l for l in loggers if l is not None]
    if not active:
        return None

    if len(active) == 1:
        return active[0]

    return CompositeLogger(active)
