from __future__ import annotations
from abc import ABC, abstractmethod

class BaseBackend(ABC):
    def __init__(self, model, logger=None):
        self.model = model
        self.logger = logger
        self.total_flop: int = 0
        self._last_batch_flop: int = 0
        self._batch_idx: int = 0
        self._epoch_idx: int = 0

    @abstractmethod
    def start(self):
        """Attach hooks"""
        ...

    @abstractmethod
    def stop(self):
        """Detaches the hook"""
        ...

    def get_total_flop(self) -> int:
        return int(self.total_flop)
