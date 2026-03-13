from __future__ import annotations
import csv
from .base_logger import BaseLogger


class CsvLogger(BaseLogger):
    def __init__(self, export_path: str | None):
        self.export_path = export_path
        self._summary_dict = None

    def log_summary(self, summary: dict):
        self._summary_dict = {
            "model FLOP": summary.get("total_model_flop", 0),
            "loss forward FLOP": summary.get("total_loss_forward_flop", 0),
            "loss backward FLOP": summary.get("total_loss_backward_flop", 0),
            "optimizer FLOP": summary.get("total_optimizer_flop", 0),
            "overall FLOP": summary.get("total_overall_flop", 0),
        }
        return self._summary_dict

    def close(self):
        if self.export_path is None or self._summary_dict is None:
            return

        with open(self.export_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])

            for key, value in self._summary_dict.items():
                if key == "loss backward FLOP" and value == 0:
                    continue
                writer.writerow([key, value])
