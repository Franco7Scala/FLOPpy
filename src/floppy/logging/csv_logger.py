from __future__ import annotations
from .base_logger import BaseLogger

import csv


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
            writer.writerow(["type", "epoch_idx", "batch_idx", "metric", "value"])
            for entry in self._batch_logs:
                epoch_idx = entry.get("epoch_idx", "")
                batch_idx = entry.get("batch_idx", "")
                for key, value in entry.items():
                    if key in ("epoch_idx", "batch_idx"):
                        continue

                    writer.writerow(["batch", epoch_idx, batch_idx, key, value])

            for entry in self._epoch_logs:
                epoch_idx = entry.get("epoch_idx", "")
                batch_idx = entry.get("batch_idx", "")
                for key, value in entry.items():
                    if key in ("epoch_idx", "batch_idx"):
                        continue

                    writer.writerow(["epoch", epoch_idx, batch_idx, key, value])

            if self._summary_dict is not None:
                for key, value in self._summary_dict.items():
                    writer.writerow(["summary", "", "", key, value])
