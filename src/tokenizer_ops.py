from __future__ import annotations
from typing import Any, Callable, Optional, Sequence
import torch

class TokenizerWithOps:
    """
   A wrapper for a tokenizer (typically HuggingFace) that estimates the cost of tokenization 
   operations and accumulates it in an internal counter.
   Objective:
   - Does not calculate FLOP, but tracks a "cost" metric for the tokenizer.
   The default cost model is:
       ops = number of characters + number of tokens generated
   Attributes:
   - base_tokenizer : callable
       The original tokenizer.
   - cost_model : str
       The name of the cost model. Currently supported: "chars+tokens".
   - total_ops : int
       The total sum of estimated operations.
   - tracker : optional
       If provided, it must expose the method add_preproc_ops(int).
    """

    def __init__(
        self,
        base_tokenizer: Callable[..., Any],
        tracker: Optional[Any] = None,
        cost_model: str = "chars+tokens",
    ) -> None:
        self.base_tokenizer = base_tokenizer
        self.tracker = tracker
        self.cost_model = cost_model
        self.total_ops: int = 0

    def __call__(self, texts: Any, *args, **kwargs) -> Any:
        """
        Allows the wrapper to be used like a standard tokenizer.
        Example:
        tokenizer = TokenizerWithOps(base_tokenizer)
            enc = tokenizer(["text 1", "text 2"], padding=True, ...)
        """
        # Normalize to estimate the number of characters.
        batch_texts = self._normalize_texts(texts)

        # The tokenizer must always receive the original input.
        enc = self.base_tokenizer(texts, *args, **kwargs)

        # Cost estimation
        ops = self._estimate_ops(batch_texts, enc)

        if ops > 0:
            self.total_ops += ops

            if self.tracker is not None and hasattr(self.tracker, "add_preproc_ops"):
                try:
                    self.tracker.add_preproc_ops(ops)
                except Exception:
                    pass

        return enc

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------

    def _normalize_texts(self, texts: Any) -> Sequence[str]:
        """
        Transforms the input into a list of strings for character count estimation. 
        If it fails to interpret the input, it returns an empty list.
        """
        if isinstance(texts, str):
            return [texts]

        if isinstance(texts, Sequence) and not isinstance(texts, (str, bytes)):
            if all(isinstance(t, str) for t in texts):
                return list(texts)

        return []

    def _estimate_ops(self, texts: Sequence[str], enc: Any) -> int:
        """
        Estimates the tokenization cost according to the selected cost_model
        - "chars+tokens" Model:
            ops = sum of text lengths + total number of tokens in input
        """
        if self.cost_model != "chars+tokens":
            return 0

        n_chars = sum(len(t) for t in texts) if texts else 0

        n_tokens = 0
        input_ids = None

        if isinstance(enc, dict):
            input_ids = enc.get("input_ids", None)
        elif hasattr(enc, "get"):
            try:
                input_ids = enc.get("input_ids", None)
            except Exception:
                input_ids = None

        if input_ids is not None:
            if isinstance(input_ids, torch.Tensor):
                n_tokens = int(input_ids.numel())

            elif isinstance(input_ids, (list, tuple)):
                if input_ids and isinstance(input_ids[0], (list, tuple)):
                    n_tokens = sum(len(seq) for seq in input_ids)
                else:
                    n_tokens = len(input_ids)

        return int(n_chars + n_tokens)


def wrap_tokenizer(
    base_tokenizer: Callable[..., Any],
    tracker: Optional[Any] = None,
    cost_model: str = "chars+tokens",
) -> TokenizerWithOps:
    """
    Helper that returns a TokenizerWithOps instance.
    Example:
        base_tok = AutoTokenizer.from_pretrained(...)
        tracked_tok = wrap_tokenizer(base_tok, tracker=tracker)
    """
    return TokenizerWithOps(
        base_tokenizer=base_tokenizer,
        tracker=tracker,
        cost_model=cost_model,
    )
