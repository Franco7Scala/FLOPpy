from __future__ import annotations
from abc import ABC, abstractmethod

class BaseLogger(ABC):
    """
    minimal Logger:
    - only recive the final summary 
    - optionally export / send
    """

    @abstractmethod
    def log_summary(self, summary: dict):
        """
        Receives the final FLOP summary and can return it
        """
        ...

    @abstractmethod
    def close(self):
        ...
