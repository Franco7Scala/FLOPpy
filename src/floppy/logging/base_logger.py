from __future__ import annotations
from abc import ABC, abstractmethod


class BaseLogger(ABC):
    """
    Minimal logger interface for FLOP/BOP summaries.
    """

    def log_batch(self, summary: dict):
        return summary

    def log_epoch(self, summary: dict):
        return summary

    @abstractmethod
    def log_summary(self, summary: dict):
        """
        Receives the final computational summary and may return it.
        """
        ...

    @abstractmethod
    def close(self):
        ...
