from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import torch


class TokenizerWithOps:
    """
    Wrapper per un tokenizer (tipicamente HuggingFace) che stima il costo
    delle operazioni di tokenizzazione e lo accumula in un contatore interno.

    Obiettivo:
        - Non calcola FLOP, ma conta una metrica di "costo" del tokenizer.
        - Il modello di costo predefinito è:
              ops = numero di caratteri + numero di token generati

    Attributi:
        base_tokenizer : callable
            Il tokenizer originale (es. AutoTokenizer.from_pretrained(...)).
        cost_model : str
            Nome del modello di costo. "chars+tokens".
        total_ops : int
            Somma di tutte le operazioni stimate effettuate dal tokenizer
            da quando il wrapper è stato creato.
        tracker : opzionale
            Se fornito, deve esporre un metodo `add_preproc_ops(int)`.
            In tal caso le operazioni stimate verranno propagate anche
            al Tracker della libreria.
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

    # ------------------------------------------------------------------
    # Metodo principale: permette di usare il wrapper come un normale tokenizer
    # ------------------------------------------------------------------
    def __call__(self, texts: Any, *args, **kwargs) -> Any:
        """
        Esempio d'uso:
            tokenizer = TokenizerWithOps(base_tokenizer)
            enc = tokenizer(["testo 1", "testo 2"], padding=True, ...)
        """

        # Normalizziamo `texts` a lista di stringhe, quando possibile.
        batch_texts = self._normalize_texts(texts)

        # Chiamata al tokenizer originale
        enc = self.base_tokenizer(batch_texts, *args, **kwargs)

        # Stima del costo
        ops = self._estimate_ops(batch_texts, enc)

        # Aggiorna contatore interno
        if ops > 0:
            self.total_ops += ops

            # Se è stato fornito un tracker compatibile, propaghiamo le ops
            if self.tracker is not None and hasattr(self.tracker, "add_preproc_ops"):
                try:
                    self.tracker.add_preproc_ops(ops)
                except Exception:
                    # un problema nel tracker non blocca la tokenizzazione
                    pass

        return enc

    # ------------------------------------------------------------------
    # Funzioni di supporto
    # ------------------------------------------------------------------
    def _normalize_texts(self, texts: Any) -> Sequence[str]:
        """
        Trasforma l'input in una lista di stringhe, per il calcolo n_char.
        Se non riusciamo a interpretarlo, lo lasciamo invariato.
        """
        # Caso singola stringa
        if isinstance(texts, str):
            return [texts]

        # Caso sequenza di stringhe
        if isinstance(texts, Sequence) and all(isinstance(t, str) for t in texts):
            return list(texts)

        return []

    def _estimate_ops(self, texts: Sequence[str], enc: Any) -> int:
        """
        Stima il "costo" di tokenizzazione secondo il cost_model selezionato.

        Modello predefinito "chars+tokens":
            ops = somma lunghezze dei testi + numero totale di token in input_ids
        """
        if self.cost_model != "chars+tokens":
           
            return 0

        if not texts:
           
            n_chars = 0
        else:
            n_chars = sum(len(t) for t in texts)

        # Cerchiamo input_ids nell'output del tokenizer
        n_tokens = 0
        input_ids = None
        if isinstance(enc, dict):
            input_ids = enc.get("input_ids", None)

        if input_ids is not None:
            # Caso tensore PyTorch
            if isinstance(input_ids, torch.Tensor):
                n_tokens = int(input_ids.numel())
            # Caso lista di liste di int
            elif isinstance(input_ids, (list, tuple)):
                if input_ids and isinstance(input_ids[0], (list, tuple)):
                    n_tokens = sum(len(seq) for seq in input_ids)
                else:
                    # lista piatta di token
                    n_tokens = len(input_ids)

        ops = n_chars + n_tokens
        return int(ops)


# ----------------------------------------------------------------------
#  Helper (opzionale)
# ----------------------------------------------------------------------

def wrap_tokenizer(
    base_tokenizer: Callable[..., Any],
    tracker: Optional[Any] = None,
    cost_model: str = "chars+tokens",
) -> TokenizerWithOps:
    """
    Helper che restituisce un TokenizerWithOps.

    Esempio:
        ft = FlopTracker(run_name="hf_exp")
        base_tok = AutoTokenizer.from_pretrained(...)
        tracked_tok = wrap_tokenizer(base_tok, tracker=None)

    Passando un tracker compatibile (es. istanza di Tracker interno),
    le operazioni verranno propagate anche a `tracker.add_preproc_ops(...)`.
    """
    return TokenizerWithOps(base_tokenizer, tracker=tracker, cost_model=cost_model)

