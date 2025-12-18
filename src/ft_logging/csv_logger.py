from __future__ import annotations
import csv
from .base_logger import BaseLogger


class CsvLogger(BaseLogger):
    def __init__(self, export_path: str | None, log_per_batch: bool, log_per_epoch: bool):
        super().__init__(log_per_batch=log_per_batch, log_per_epoch=log_per_epoch)
        self.export_path = export_path
        self._entries_batch = []
        self._entries_epoch = []

    def log_batch(self, step, flop, cumulative_flop, epoch=None):
        if not self.log_per_batch:
            return
        self._entries_batch.append(
            {
                "type": "batch",
                "step": step,
                "epoch": epoch,
                "flop": flop,
                "cumulative_flop": cumulative_flop,
            }
        )

    def log_epoch(self, epoch, flop, cumulative_flop):
        if not self.log_per_epoch:
            return
        self._entries_epoch.append(
            {
                "type": "epoch",
                "epoch": epoch,
                "flop": flop,
                "cumulative_flop": cumulative_flop,
            }
        )

    def close(self):
        if self.export_path is None:
            return

        fieldnames = ["type", "step", "epoch", "flop", "cumulative_flop"]
        with open(self.export_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for e in self._entries_batch + self._entries_epoch:
                writer.writerow(e)
