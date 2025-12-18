from __future__ import annotations
from .csv_logger import CsvLogger
from .wandb_logger import WandbLogger


def create_logger(
    log_per_batch: bool,
    log_per_epoch: bool,
    export_path: str | None,
    use_wandb: bool,
    wandb_project: str | None,
    wandb_token: str | None,
    run_name: str | None,
):
    loggers = []

    if export_path is not None or log_per_batch or log_per_epoch:
        loggers.append(
            CsvLogger(
                export_path=export_path,
                log_per_batch=log_per_batch,
                log_per_epoch=log_per_epoch,
            )
        )

    if use_wandb:
        loggers.append(
            WandbLogger(
                project=wandb_project,
                token=wandb_token,
                log_per_batch=log_per_batch,
                log_per_epoch=log_per_epoch,
                run_name=run_name,
            )
        )

    if not loggers:
        return None
    if len(loggers) == 1:
        return loggers[0]

    return MultiplexLogger(loggers)


class MultiplexLogger:
    def __init__(self, loggers):
        self.loggers = loggers

    def log_batch(self, *args, **kwargs):
        for lg in self.loggers:
            lg.log_batch(*args, **kwargs)

    def log_epoch(self, *args, **kwargs):
        for lg in self.loggers:
            lg.log_epoch(*args, **kwargs)

    def close(self):
        for lg in self.loggers:
            lg.close()
