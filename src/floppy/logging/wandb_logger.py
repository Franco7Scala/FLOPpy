from __future__ import annotations
from .base_logger import BaseLogger
from ..utils.utility import cprint, Color

import wandb
import os


class WandbLogger(BaseLogger):

    def __init__(self, reporter_key: str, project_name: str | None = None, group_name: str | None = None, run_name: str | None = None):
        self._run = None
        os.environ["WANDB_SILENT"] = "true"
        os.environ["WANDB_CONSOLE"] = "off"
        wandb.login(key=reporter_key)
        try:
            self._run = wandb.init(entity=group_name, project=project_name, name=run_name, settings=wandb.Settings(quiet=True, silent=True))

        except Exception as e:
            self._run = None
            error_msg = str(e).split(":")[-1].strip() if ":" in str(e) else str(e)
            if "401" in str(e):
                error_msg = "Invalid API Key or Unauthorized (401)"

            cprint(f"Failed to initialize Weights & Biases logger: {error_msg}", Color.WARNING)

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
        summary_dict = dict(summary)
        if self._run is not None:
            payload = dict(summary_dict)
            payload["log_type"] = "summary"
            wandb.log(payload)
            self._run.summary.update(summary_dict)

        return summary_dict

    def close(self):
        if self._run is not None:
            self._run.finish()
