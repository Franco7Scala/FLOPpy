from __future__ import annotations
from .base_logger import BaseLogger


class WandbLogger(BaseLogger):
    def __init__(
        self,
        use_wandb: bool,
        wandb_project: str | None = None,
        wandb_token: str | None = None,
        run_name: str | None = None,
    ):
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_token = wandb_token
        self.run_name = run_name
        self._summary_dict = None
        self._wandb = None
        self._run = None

        if not self.use_wandb:
            return

        try:
            import wandb
            self._wandb = wandb

            if self.wandb_token:
                try:
                    wandb.login(key=self.wandb_token)
                except Exception:
                    pass

            self._run = wandb.init(
                project=self.wandb_project or "floppy",
                name=self.run_name,
                reinit=True,
            )
        except Exception:
            self._wandb = None
            self._run = None

    def log_batch(self, summary: dict):
        if self._run is not None:
            try:
                payload = dict(summary)
                payload["log_type"] = "batch"
                self._wandb.log(payload)
            except Exception:
                pass
        return summary

    def log_epoch(self, summary: dict):
        if self._run is not None:
            try:
                payload = dict(summary)
                payload["log_type"] = "epoch"
                self._wandb.log(payload)
            except Exception:
                pass
        return summary

    def log_summary(self, summary: dict):
        self._summary_dict = dict(summary)

        if self._run is not None:
            try:
                self._wandb.log(self._summary_dict)
                self._run.summary.update(self._summary_dict)
            except Exception:
                pass

        return self._summary_dict

    def close(self):
        if self._run is not None:
            try:
                self._run.finish()
            except Exception:
                pass
