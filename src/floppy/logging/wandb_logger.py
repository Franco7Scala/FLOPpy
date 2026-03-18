from __future__ import annotations
from .base_logger import BaseLogger

import wandb


class WandbLogger(BaseLogger):
    """
    Logger for Weights & Biases.
    Only records the final FLOP summary.
    """

    def __init__(self, wandb_project_name: str, wandb_token: str, wandb_run_name: str):
        self.wandb_project_name = wandb_project_name
        self.wandb_token = wandb_token
        self.wandb_run_name = wandb_run_name
        wandb.login(key=self.wandb_token)
        self._run = wandb.init(project=self.wandb_project_name, name=self.wandb_run_name, reinit=True)

    def log_summary(self, summary: dict):
        summary_dict = {
            "total_model_flops": summary.get("total_model_flop", 0),
            "total_optimizer_flops": summary.get("total_optimizer_flop", 0),
            "total_loss_forward_flops": summary.get("total_loss_forward_flop", 0),
            "total_loss_backward_flops": summary.get("total_loss_backward_flop", 0),
            "total_overall_flops": summary.get("total_overall_flop", 0),
        }
        if self._run is not None:
            wandb.log(summary_dict)
            self._run.summary.update(summary_dict)

        return summary_dict

    def close(self):
        if self._run is not None:
            self._run.finish()
            self._run = None
