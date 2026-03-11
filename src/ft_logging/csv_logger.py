from __future__ import annotations
import csv
from .base_logger import BaseLogger

class CsvLogger(BaseLogger):
    def __init__(self, export_path: str | None):
        self.export_path = export_path
        self._summary_dict = None

    def log_summary(self, summary: dict):
        self._summary_dict = {
            "total_model_flop": summary.get("total_model_flop", 0),
            "total_optimizer_flop": summary.get("total_optimizer_flop", 0),
            "total_loss_forward_flop": summary.get("total_loss_forward_flop", 0),
            "total_loss_backward_flop": summary.get("total_loss_backward_flop", 0),
            "total_overall_flop": summary.get("total_overall_flop", 0),
        }
        return self._summary_dict

    def close(self):
        if self.export_path is None or self._summary_dict is None:
            return

        with open(self.export_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "total_model_flop",
                    "total_optimizer_flop",
                    "total_loss_forward_flop",
                    "total_loss_backward_flop",
                    "total_overall_flop",
                ],
            )
            writer.writeheader()
            writer.writerow(self._summary_dict)
