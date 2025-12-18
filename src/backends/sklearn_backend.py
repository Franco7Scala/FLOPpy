from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .base import BaseBackend


class SklearnBackend(BaseBackend):
    """
    Backend per modelli scikit-learn.
    - Ogni chiamata conta i FLOP in base al tipo di modello + shape di X.
    - Ogni chiamata viene trattata come un "batch" nei log.
    """

    def __init__(self, model, logger=None):
        super().__init__(model, logger=logger)
        self._orig_fit: Callable | None = None
        self._orig_predict: Callable | None = None
        self._orig_predict_proba: Callable | None = None
        self._orig_transform: Callable | None = None

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
            self.model.predict_proba = self._wrap_predict(self.model.predict_proba)

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

    # ---------------- WRAPPER METODI ---------------- #

    def _wrap_fit(self, fn: Callable) -> Callable:
        def wrapped(X, y=None, *args, **kwargs):
            result = fn(X, y, *args, **kwargs)
            # stimiamo i FLOP di training
            flop = self._estimate_fit_flop(np.asarray(X), y)
            self._accumulate_call(flop)
            return result

        return wrapped

    def _wrap_predict(self, fn: Callable) -> Callable:
        def wrapped(X, *args, **kwargs):
            X_arr = np.asarray(X)
            y = fn(X, *args, **kwargs)
            flop = self._estimate_predict_flop(X_arr, np.asarray(y))
            self._accumulate_call(flop)
            return y

        return wrapped

    def _wrap_transform(self, fn: Callable) -> Callable:
        def wrapped(X, *args, **kwargs):
            X_arr = np.asarray(X)
            Z = fn(X, *args, **kwargs)
            flop = self._estimate_transform_flop(X_arr, np.asarray(Z))
            self._accumulate_call(flop)
            return Z

        return wrapped

    # ---------------- ACCUMULO E LOG ---------------- #

    def _accumulate_call(self, flop: int):
        self._last_batch_flop = int(flop)
        self.total_flop += int(flop)
        self._batch_idx += 1

        if self.logger is not None and hasattr(self.logger, "log_batch"):
            self.logger.log_batch(
                step=self._batch_idx,
                flop=self._last_batch_flop,
                cumulative_flop=self.total_flop,
                epoch=self._epoch_idx,
            )

    # ---------------- STIME FLOP ---------------- #

    def _estimate_fit_flop(self, X: np.ndarray, y: Any) -> int:
        """
        Per ora teniamo fit() come 0 FLOP (o molto grezzo).
        Eventualmente puoi estendere con formule specifiche per algoritmo.
        """
        return 0

    def _estimate_predict_flop(self, X: np.ndarray, y: np.ndarray) -> int:
        """
        Stima dei FLOP per una chiamata a predict(X).
        Alcune stime sono molto approssimate ma sufficienti per confronto.
        """
        try:
            from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
            from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        except ImportError:
            return 0

        n_samples, n_features = X.shape
        n_outputs = 1 if y.ndim == 1 else y.shape[1]

        m = self.model

        # ------- Regressione / classificazione lineare ------- #
        if isinstance(m, (LinearRegression, Ridge, Lasso, LogisticRegression)):
            # y = XW^T + b 
            flop = 2 * n_features * n_outputs * n_samples
            return int(flop)

        # ------- KNN ------- #
        if isinstance(m, (KNeighborsClassifier, KNeighborsRegressor)):
            # brute force: distanza vs tutti i punti di training:
            # ~ 2 * n_train * n_features * n_test
            n_train = getattr(m, "n_samples_fit_", None)
            if n_train is None and hasattr(m, "_fit_X"):
                n_train = m._fit_X.shape[0]
            if n_train is None:
                return 0
            flop = 2 * n_train * n_features * n_samples
            return int(flop)
        return 0

    def _estimate_transform_flop(self, X: np.ndarray, Z: np.ndarray) -> int:
        """
        Per trasformatori sklearn (StandardScaler, PCA, ecc.).
        Per ora lo lasciamo a 0, ma puoi estenderlo facilmente.
        """
        return 0
