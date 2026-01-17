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
        """Aggancia gli hook"""
        ...

    @abstractmethod
    def stop(self):
        """Rimuove gli hook"""
        ...

    def add_extra_flop(self, flop: int) -> None:
        """
        Aggiunge FLOP calcolati esternamente agli hook (es. loss, preprocessing).
        """
        return

    def get_total_flop(self) -> int:
        return int(self.total_flop)

    def get_last_batch_flop(self) -> int:
        return int(self._last_batch_flop)

    def set_epoch(self, epoch: int):
        """Opzionale: permette di tracciare l'epoch corrente nei log."""
        self._epoch_idx = epoch
