from __future__ import annotations

from .base import BaseBackend


def create_backend(model, backend: str, logger=None) -> BaseBackend:
    """
    Factory with lazy import:
    - Does not import torch_backend / hf_backend at import-time;
    - Sklearn works even without torch/transformers installed;
    - Prevents crashes if torch_backend has errors while testing sklearn.
    """
    backend = (backend or "auto").lower()

    # ---------- PyTorch ---------- #
    if backend in ("auto"):
        try:
            import torch
            from torch.nn import Module
            if isinstance(model, Module):
                from .torch_backend import TorchBackend  # lazy import
                return TorchBackend(model, logger=logger)
        except ImportError:
            if backend == "torch":
                raise

    # ---------- Sklearn ---------- #
    if backend in ("auto"):
        try:
            from sklearn.base import BaseEstimator
            if isinstance(model, BaseEstimator):
                from .sklearn_backend import SklearnBackend  # lazy import
                return SklearnBackend(model, logger=logger)
        except ImportError:
            if backend == "sklearn":
                raise

    raise ValueError(f"Unable to determine backend for model: {type(model)} (backend={backend})")
