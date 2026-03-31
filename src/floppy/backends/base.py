from __future__ import annotations
from abc import ABC, abstractmethod


class BaseBackend(ABC):
    """
    Abstract backend for model FLOP counting.

    Responsibilities:
    - attach / detach framework-specific hooks or wrappers
    - maintain total model FLOP (Forward and Backward)
    - expose aggregate model FLOP
    """
    def __init__(self, model, logger=None):
        self.model = model
        self.logger = logger
        # --- Forward metrics ---
        self.total_forward_flop: int = 0
        self.total_forward_bop: int = 0
        self._last_batch_flop: int = 0
        self._last_batch_bop: int = 0
        # --- Backward metrics ---
        self.total_backward_flop: int = 0
        self.total_backward_bop: int = 0
        self._last_batch_backward_flop: int = 0
        self._last_batch_backward_bop: int = 0
        self._batch_idx: int = 0

    @abstractmethod
    def start(self):
        """Attach hooks / wrappers used to count model FLOP."""
        ...

    @abstractmethod
    def stop(self):
        """Detach hooks / wrappers used to count model FLOP."""
        ...

    # ------------------------------------------------------------
    # Forward Getters
    # ------------------------------------------------------------

    def get_total_forward_flop(self) -> int:
        return int(self.total_forward_flop)

    def get_last_batch_flop(self) -> int:
        return int(self._last_batch_flop)

    def get_total_forward_bop(self) -> int:
        return int(self.total_forward_bop)

    def get_last_batch_bop(self) -> int:
        return int(self._last_batch_bop)

    # ------------------------------------------------------------
    # Backward Getters
    # ------------------------------------------------------------

    def get_total_backward_flop(self) -> int:
        return int(self.total_backward_flop)

    def get_total_backward_bop(self) -> int:
        return int(self.total_backward_bop)
