from __future__ import annotations
from .base_logger import BaseLogger

import wandb


class WandbLogger(BaseLogger):

    def __init__(self, wandb_token: str, wandb_project: str | None = None, run_name: str | None = None):
        self.wandb_project = wandb_project
        self.wandb_token = wandb_token
        self.run_name = run_name
        self._summary_dict = None
        self._run = None
        wandb.login(key=self.wandb_token)
        self._run = wandb.init(project=self.wandb_project or "floppy", name=self.run_name, reinit=True)


    def log_batch(self, summary: dict):
        if self._run is not None:
            payload = dict(summary)
            payload["log_type"] = "batch"
            wandb.log(payload)

        return summary

    def log_epoch(self, summary: dict):
        if self._run is not None:
            payload = dict(summary)
            payload["log_type"] = "epoch"
            wandb.log(payload)

        return summary

    def log_summary(self, summary: dict):
        self._summary_dict = dict(summary)
        if self._run is not None:
            payload = dict(self._summary_dict)
            payload["log_type"] = "summary"
            wandb.log(payload)
            self._run.summary.update(self._summary_dict)

        return self._summary_dict

    def close(self):
        if self._run is not None:
            self._run.finish()
