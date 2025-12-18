from __future__ import annotations
from .base_logger import BaseLogger


class WandbLogger(BaseLogger):
    def __init__(
        self,
        project: str | None,
        token: str | None,
        log_per_batch: bool,
        log_per_epoch: bool,
        run_name: str | None = None,
    ):
        super().__init__(log_per_batch=log_per_batch, log_per_epoch=log_per_epoch)
        self._wandb = None
        self.run = None

        if project is not None:
            import wandb
            if token is not None:
                wandb.login(key=token)
            self._wandb = wandb
            self.run = wandb.init(project=project, name=run_name)

    def log_batch(self, step, flop, cumulative_flop, epoch=None):
        if not self.log_per_batch or self._wandb is None:
            return
        self._wandb.log(
            {
                "flop_batch": flop,
                "flop_cumulative": cumulative_flop,
                "batch_step": step,
                "epoch": epoch,
            }
        )

    def log_epoch(self, epoch, flop, cumulative_flop):
        if not self.log_per_epoch or self._wandb is None:
            return
        self._wandb.log(
            {
                "flop_epoch": flop,
                "flop_cumulative": cumulative_flop,
                "epoch": epoch,
            }
        )

    def close(self):
        if self.run is not None:
            self.run.finish()
