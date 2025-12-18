from __future__ import annotations

import torch.nn as nn

from .torch_backend import TorchBackend

try:
    from transformers import PreTrainedModel
except ImportError:
    # se transformers non è installato, definiamo un placeholder
    PreTrainedModel = nn.Module


class HFBackend(TorchBackend):
    """
    Backend per modelli HuggingFace (transformers.PreTrainedModel).

    Estende TorchBackend, riusando gli stessi hook.
    """

    def __init__(self, model: PreTrainedModel, logger=None):
        super().__init__(model, logger=logger)
