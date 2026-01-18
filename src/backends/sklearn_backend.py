from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from .base import BaseBackend


class SklearnBackend(BaseBackend):
    """
    Backend per modelli scikit-learn.
    - Ogni chiamata conta i FLOP in base al tipo di modello + shape di X.
    - Ogni chiamata viene trattata come un "batch" nei log.
    - Supporta FLOP extra (loss/preproc) via add_extra_flop().
    """

    def __init__(self, model, logger=None):
        super().__init__(model, logger=logger)
        self._orig_fit: Optional[Callable] = None
        self._orig_predict: Optional[Callable] = None
        self._orig_predict_proba: Optional[Callable] = None
        self._orig_transform: Optional[Callable] = None

        # FLOP esterni da accumulare nella prossima call
        self._pending_extra_flop: int = 0

    # ---------------- START / STOP ---------------- #

    def start(self):
        if hasattr(self.model, "fit"):
            self._orig_fit = self.model.fit
            self.model.fit = self._wrap_fit(self.model.fit)

        if hasattr(self.model, "predict"):
            self._orig_predict = self.model.predict
            self.model.predict = self._wrap_predict(self.model.predict)

        if hasattr(self.model, "predict_proba"):
            self._orig_predict_proba = self.model.predict_proba
            self.model.predict_proba = self._wrap_predict_proba(self.model.predict_proba)

        if hasattr(self.model, "transform"):
            self._orig_transform = self.model.transform
            self.model.transform = self._wrap_transform(self.model.transform)

    def stop(self):
        if self._orig_fit is not None:
            self.model.fit = self._orig_fit
        if self._orig_predict is not None:
            self.model.predict = self._orig_predict
        if self._orig_predict_proba is not None:
            self.model.predict_proba = self._orig_predict_proba
        if self._orig_transform is not None:
            self.model.transform = self._orig_transform

    # ---------------- EXTRA FLOP ---------------- #

    def add_extra_flop(self, flop: int) -> None:
        """
        Aggiunge FLOP esterni (loss/preproc) alla prossima call (fit/predict/transform).
        """
        if flop is None:
            return
        v = int(flop)
        if v > 0:
            self._pending_extra_flop += v

    def _consume_extra(self) -> int:
        v = int(self._pending_extra_flop)
        self._pending_extra_flop = 0
        return v

    # ---------------- WRAPPER METODI ---------------- #

    def _wrap_fit(self, fn: Callable) -> Callable:
        def wrapped(X, y=None, *args, **kwargs):
            result = fn(X, y, *args, **kwargs)
            flop = self._estimate_fit_flop(np.asarray(X), y)
            flop += self._consume_extra()
            self._accumulate_call(flop)
            return result

        return wrapped

    def _wrap_predict(self, fn: Callable) -> Callable:
        def wrapped(X, *args, **kwargs):
            X_arr = np.asarray(X)
            y_pred = fn(X, *args, **kwargs)
            flop = self._estimate_predict_flop(X_arr, np.asarray(y_pred))
            flop += self._consume_extra()
            self._accumulate_call(flop)
            return y_pred

        return wrapped

    def _wrap_predict_proba(self, fn: Callable) -> Callable:
        def wrapped(X, *args, **kwargs):
            X_arr = np.asarray(X)
            proba = fn(X, *args, **kwargs)
            # per semplicità uso la stessa stima di predict 
            flop = self._estimate_predict_flop(X_arr, np.asarray(proba))
            flop += self._consume_extra()
            self._accumulate_call(flop)
            return proba

        return wrapped

    def _wrap_transform(self, fn: Callable) -> Callable:
        def wrapped(X, *args, **kwargs):
            X_arr = np.asarray(X)
            Z = fn(X, *args, **kwargs)
            flop = self._estimate_transform_flop(X_arr, np.asarray(Z))
            flop += self._consume_extra()
            self._accumulate_call(flop)
            return Z

        return wrapped

    # ---------------- ACCUMULO E LOG ---------------- #

    def _accumulate_call(self, flop: int):
        self._last_batch_flop = int(flop)
        self.total_flop += int(flop)
        self._batch_idx += 1

        if self.logger is not None and hasattr(self.logger, "log_batch") and getattr(self.logger, "log_per_batch", True):
            self.logger.log_batch(
                step=self._batch_idx,
                flop=self._last_batch_flop,
                cumulative_flop=self.total_flop,
                epoch=self._epoch_idx,
            )

    # ---------------- STIME FLOP ---------------- #

    def _estimate_fit_flop(self, X: np.ndarray, y: Any) -> int:
        return 0

    def _estimate_predict_flop(self, X: np.ndarray, y: np.ndarray) -> int:
        try:
            from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
            from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        except ImportError:
            return 0

        n_samples, n_features = X.shape
        n_outputs = 1 if y.ndim == 1 else y.shape[1]

        m = self.model

        # Lineari / logreg: XW + b
        if isinstance(m, (LinearRegression, Ridge, Lasso, LogisticRegression)):
            flop = 2 * n_features * n_outputs * n_samples
            return int(flop)

        # KNN brute force: distanze vs training
        if isinstance(m, (KNeighborsClassifier, KNeighborsRegressor)):
            n_train = getattr(m, "n_samples_fit_", None)
            if n_train is None and hasattr(m, "_fit_X"):
                n_train = m._fit_X.shape[0]
            if n_train is None:
                return 0
            flop = 2 * int(n_train) * n_features * n_samples
            return int(flop)

        return 0

    def _estimate_transform_flop(self, X: np.ndarray, Z: np.ndarray) -> int:
        return 0
