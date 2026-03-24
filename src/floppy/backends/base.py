from __future__ import annotations
from abc import ABC, abstractmethod


class BaseBackend(ABC):
    """
    Abstract backend for model FLOP counting.

    Responsibilities:
    - attach / detach framework-specific hooks or wrappers
    - maintain total model FLOP
    - expose aggregate model FLOP

    Notes:
    - only model FLOP belong here
    - loss / optimizer / preprocessing FLOP are tracked elsewhere
    """

    def __init__(self, model, logger=None):
        self.model = model
        self.logger = logger
        self.total_flop: int = 0
        self.total_bop: int = 0
        self._last_batch_flop: int = 0
        self._last_batch_bop: int = 0
        self._batch_idx: int = 0

    @abstractmethod
    def start(self):
        """Attach hooks / wrappers used to count model FLOP."""
        ...

    @abstractmethod
    def stop(self):
        """Detach hooks / wrappers used to count model FLOP."""
        ...

    def get_total_flop(self) -> int:
        return int(self.total_flop)

    def get_last_batch_flop(self) -> int:
        return int(self._last_batch_flop)

    def get_total_bop(self) -> int:
        return int(self.total_bop)

    def get_last_batch_bop(self) -> int:
        return int(self._last_batch_bop)
