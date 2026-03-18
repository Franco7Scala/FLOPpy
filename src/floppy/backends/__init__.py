from __future__ import annotations
from .base import BaseBackend
from torch.nn import Module
from sklearn.base import BaseEstimator

import torch


def create_backend(model, backend: str = "auto", logger=None) -> BaseBackend:
    """
    Backend factory with lazy imports.

    Goals:
    - avoid importing torch/sklearn backend modules at import-time
    - keep the library lightweight and robust
    - support backend auto-detection
    """

    backend = (backend or "auto").lower()

    # ---------- PyTorch ---------- #
    if backend in ("auto", "torch"):
        if isinstance(model, Module):
            from .torch_backend import TorchBackend
            return TorchBackend(model, logger=logger)

    # ---------- Scikit-learn ---------- #
    if backend in ("auto", "sklearn"):
        if isinstance(model, BaseEstimator):
            from .sklearn_backend import SklearnBackend
            return SklearnBackend(model, logger=logger)

    raise ValueError(f"Unable to determine backend for model type {type(model)} with backend='{backend}'!")
