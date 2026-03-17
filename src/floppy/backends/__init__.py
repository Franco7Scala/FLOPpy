from __future__ import annotations
from .base import BaseBackend


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
        try:
            import torch
            from torch.nn import Module
            if isinstance(model, Module):
                from .torch_backend import TorchBackend
                return TorchBackend(model, logger=logger)

        except ImportError:
            if backend == "torch":
                raise

    # ---------- Scikit-learn ---------- #
    if backend in ("auto", "sklearn"):
        try:
            from sklearn.base import BaseEstimator
            if isinstance(model, BaseEstimator):
                from .sklearn_backend import SklearnBackend
                return SklearnBackend(model, logger=logger)

        except ImportError:
            if backend == "sklearn":
                raise

    raise ValueError(
        f"Unable to determine backend for model type {type(model)} with backend='{backend}'."
    )
