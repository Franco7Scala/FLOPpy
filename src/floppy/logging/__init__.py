from __future__ import annotations
from .csv_logger import CsvLogger
from .wandb_logger import WandbLogger
from ..utils.wandb_configuration import WandbConfiguration


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
    wandb_config: WandbConfiguration | None = None,
    run_name: str | None = None,
):
    loggers = []
    if export_path is not None:
        loggers.append(CsvLogger(export_path=export_path))

    if wandb_config is not None:
        loggers.append(WandbLogger(project_name=wandb_config.project_name, group_name=wandb_config.group_name, reporter_key=wandb_config.reporter_key, run_name=run_name))

    active = [l for l in loggers if l is not None]
    if not active:
        return None

    if len(active) == 1:
        return active[0]

    return CompositeLogger(active)
