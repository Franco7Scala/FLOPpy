from __future__ import annotations
from typing import Any, Callable, Sequence

import torch


class TokenizerWithOps:
    """
    Wrapper per un tokenizer HuggingFace (o simile) che conta il "costo"
    delle operazioni di tokenizzazione e le registra nel Tracker.

    Il costo è approssimato come:
      - numero di caratteri processati + numero di token prodotti
    """

    def __init__(self, base_tokenizer: Callable, tracker, cost_model: str = "chars+tokens"):
        self.base_tokenizer = base_tokenizer
        self.tracker = tracker
        self.cost_model = cost_model

    def __call__(self, texts: Any, *args, **kwargs):
        """
        Permette di usare il wrapper come un normale tokenizer:
        enc = tokenizer(texts, ...)
        """
        # Normalizziamo texts a lista di stringhe
        if isinstance(texts, str):
            batch_texts = [texts]
        elif isinstance(texts, Sequence):
            batch_texts = list(texts)
        else:
            return self.base_tokenizer(texts, *args, **kwargs)

        # Chiamo il tokenizer vero
        enc = self.base_tokenizer(batch_texts, *args, **kwargs)

        # Stimo il costo
        if self.cost_model == "chars+tokens":
            n_chars = sum(len(t) for t in batch_texts)
            n_tokens = 0

            # supporta sia tensori PyTorch che liste
            input_ids = enc.get("input_ids", None)
            if input_ids is not None:
                if isinstance(input_ids, torch.Tensor):
                    n_tokens = int(input_ids.numel())
                elif isinstance(input_ids, (list, tuple)):
                    n_tokens = sum(len(seq) for seq in input_ids)

            ops = n_chars + n_tokens
            self.tracker.add_preproc_ops(ops)

        return enc
