from __future__ import annotations
from .base import BaseBackend
from .torch_backend import TorchBackend
from .sklearn_backend import SklearnBackend
from .hf_backend import HFBackend

try:
    from transformers import PreTrainedModel
except ImportError:
    PreTrainedModel = None


def create_backend(model, backend: str, logger=None) -> BaseBackend:
    """
    Se backend='auto' prova a riconoscere il tipo di modello.
    """

    # ---------- HuggingFace ---------- #
    if backend in ("hf", "auto") and PreTrainedModel is not None:
        if isinstance(model, PreTrainedModel):
            return HFBackend(model, logger=logger)

    # ---------- PyTorch ---------- #
    if backend in ("torch", "auto"):
        try:
            import torch
            from torch.nn import Module
            if isinstance(model, Module):
                return TorchBackend(model, logger=logger)
        except ImportError:
            if backend == "torch":
                raise

    # ---------- Sklearn ---------- #
    if backend in ("sklearn", "auto"):
        try:
            from sklearn.base import BaseEstimator
            if isinstance(model, BaseEstimator):
                return SklearnBackend(model, logger=logger)
        except ImportError:
            if backend == "sklearn":
                raise

    raise ValueError(f"Impossibile determinare il backend per il modello: {type(model)}")
