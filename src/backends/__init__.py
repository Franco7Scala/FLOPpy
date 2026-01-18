from __future__ import annotations

from .base import BaseBackend

try:
    from transformers import PreTrainedModel
except Exception:
    PreTrainedModel = None


def _looks_like_hf_model(model) -> bool:
    # fallback robusto: attributi tipici di transformers
    return hasattr(model, "config") and hasattr(model, "forward")


def create_backend(model, backend: str, logger=None) -> BaseBackend:
    """
    Factory con lazy import:
    - non importa torch_backend / hf_backend a import-time
    - sklearn funziona anche senza torch/transformers installati
    - evita crash se torch_backend ha errori mentre stai testando sklearn
    """
    backend = (backend or "auto").lower()

    # ---------- HuggingFace ---------- #
    if backend in ("hf", "auto"):
        is_hf = False
        if PreTrainedModel is not None and isinstance(model, PreTrainedModel):
            is_hf = True
        elif _looks_like_hf_model(model):
            is_hf = True

        if is_hf:
            from .hf_backend import HFBackend  # lazy import
            return HFBackend(model, logger=logger)

        if backend == "hf":
            raise ValueError("backend='hf' ma il modello non sembra un PreTrainedModel HuggingFace.")

    # ---------- PyTorch ---------- #
    if backend in ("torch", "auto"):
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
    if backend in ("sklearn", "auto"):
        try:
            from sklearn.base import BaseEstimator
            if isinstance(model, BaseEstimator):
                from .sklearn_backend import SklearnBackend  # lazy import
                return SklearnBackend(model, logger=logger)
        except ImportError:
            if backend == "sklearn":
                raise

    raise ValueError(f"Impossibile determinare il backend per il modello: {type(model)} (backend={backend})")
