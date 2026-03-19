from __future__ import annotations
import csv
from .base_logger import BaseLogger


class CsvLogger(BaseLogger):
    def __init__(self, export_path: str | None):
        self.export_path = export_path
        self._summary_dict = None
        self._batch_logs = []
        self._epoch_logs = []

    def log_batch(self, summary: dict):
        self._batch_logs.append(dict(summary))
        return summary

    def log_epoch(self, summary: dict):
        self._epoch_logs.append(dict(summary))
        return summary

    def log_summary(self, summary: dict):
        self._summary_dict = dict(summary)
        return self._summary_dict

    def close(self):
        if self.export_path is None:
            return

        with open(self.export_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["type", "index", "metric", "value"])

            for entry in self._batch_logs:
                idx = entry.get("batch_idx", "")
                for key, value in entry.items():
                    if key == "batch_idx":
                        continue
                    writer.writerow(["batch", idx, key, value])

            for entry in self._epoch_logs:
                idx = entry.get("epoch_idx", "")
                for key, value in entry.items():
                    if key == "epoch_idx":
                        continue
                    writer.writerow(["epoch", idx, key, value])

            if self._summary_dict is not None:
                for key, value in self._summary_dict.items():
                    writer.writerow(["summary", "", key, value])
