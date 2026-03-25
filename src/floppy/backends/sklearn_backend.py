from __future__ import annotations
from typing import Any, Callable, Optional
from .base import BaseBackend
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression, SGDClassifier, SGDRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.decomposition import PCA

import numpy as np


class SklearnBackend(BaseBackend):
    """
    Backend for scikit-learn models.

    Responsibilities:
    - wraps sklearn methods (fit, predict, predict_proba, transform)
    - estimates model FLOPs
    """

    def __init__(self, model, logger=None):
        super().__init__(model, logger=logger)
        self._orig_fit: Optional[Callable] = None
        self._orig_predict: Optional[Callable] = None
        self._orig_predict_proba: Optional[Callable] = None
        self._orig_transform: Optional[Callable] = None

    # ------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Wrapped methods
    # ------------------------------------------------------------

    def _wrap_fit(self, fn: Callable) -> Callable:
        def wrapped(X, y=None, *args, **kwargs):
            result = fn(X, y, *args, **kwargs)
            x_arr = np.asarray(X)
            flop = self._estimate_fit_flop(x_arr, y)

            # Calculate BOPs based on input data precision
            bit_width = self._get_numpy_bit_width(x_arr)
            bops = int(flop * bit_width)

            self._accumulate_call(flop, bops)
            return result

        return wrapped

    def _wrap_predict(self, fn: Callable) -> Callable:
        def wrapped(X, *args, **kwargs):
            x_arr = np.asarray(X)
            y_pred = fn(X, *args, **kwargs)
            flop = self._estimate_predict_flop(x_arr, np.asarray(y_pred))

            # Calculate BOPs based on input data precision
            bit_width = self._get_numpy_bit_width(x_arr)
            bops = int(flop * bit_width)

            self._accumulate_call(flop, bops)
            return y_pred

        return wrapped

    def _wrap_predict_proba(self, fn: Callable) -> Callable:
        def wrapped(X, *args, **kwargs):
            x_arr = np.asarray(X)
            proba = fn(X, *args, **kwargs)
            flop = self._estimate_predict_flop(x_arr, np.asarray(proba))

            # Calculate BOPs based on input data precision
            bit_width = self._get_numpy_bit_width(x_arr)
            bops = int(flop * bit_width)

            self._accumulate_call(flop, bops)
            return proba

        return wrapped

    def _wrap_transform(self, fn: Callable) -> Callable:
        def wrapped(X, *args, **kwargs):
            x_arr = np.asarray(X)
            z = fn(X, *args, **kwargs)
            flop = self._estimate_transform_flop(x_arr, np.asarray(z))

            # Calculate BOPs based on input data precision
            bit_width = self._get_numpy_bit_width(x_arr)
            bops = int(flop * bit_width)

            self._accumulate_call(flop, bops)
            return z

        return wrapped

    # ------------------------------------------------------------
    # FLOP accumulation
    # ------------------------------------------------------------

    def _accumulate_call(self, flop: int, bop: int):
        self._batch_idx += 1
        value_flop = int(flop)
        value_bop = int(bop)
        self._last_batch_flop = value_flop
        self._last_batch_bop = value_bop
        self.total_flop += value_flop
        self.total_bop += value_bop

    def _get_numpy_bit_width(self, x_arr: np.ndarray) -> int:
        """
        Determines the bit-width of the NumPy array used for computation.
        Scikit-learn predominantly uses float64 internally, but tracking the
        input dtype provides a baseline for hardware effort estimation.
        """
        if x_arr is None or not hasattr(x_arr, "dtype"):
            return 64  # sklearn defaults to float64

        dtype = x_arr.dtype
        if dtype in (np.float64, np.int64, np.complex128):
            return 64

        elif dtype in (np.float32, np.int32, np.complex64):
            return 32

        elif dtype in (np.float16, np.int16):
            return 16

        elif dtype in (np.int8, np.uint8):
            return 8

        return 64  # Default fallback

    # ------------------------------------------------------------
    # Iteration helpers
    # ------------------------------------------------------------

    def _fallback_iterations(self, n_samples: int) -> int:
        """
        Dynamic fallback when no reliable iteration information is available.
        Uses a simple dataset-dependent heuristic instead of a fixed constant.
        """
        return max(10, int(np.log2(max(n_samples, 2))) * 10)

    def _resolve_effective_iterations(self, model, n_samples: int) -> int:
        """
        Resolve the most reliable available estimate for the number of training iterations.

        Priority:
        1. post-fit n_iter_
        2. post-fit t_ (mainly for SGD)
        3. dynamic fallback based on dataset size
        """
        # 1) Real post-fit iteration count 
        iters = getattr(model, "n_iter_", None)
        if iters is not None:
            if isinstance(iters, (list, tuple, np.ndarray)):
                return max(1, int(np.max(iters)))

            else:
               return max(1, int(iters))

        # 2) SGD-style total update counter
        t_value = getattr(model, "t_", None)
        if t_value is not None:
            return max(1, int(t_value))

        # dynamic fallback
        return self._fallback_iterations(n_samples)

    # ------------------------------------------------------------
    # Tree helpers
    # ------------------------------------------------------------

    def _resolve_tree_depth(self, model, n_samples: int) -> int:
        """
        Resolve a reasonable depth estimate for tree-based models.
        """
        if hasattr(model, "get_depth"):
            return max(1, int(model.get_depth()))

        return max(1, int(np.log2(max(n_samples, 2))))

    def _resolve_forest_avg_depth(self, model, n_samples: int) -> int:
        """
        Resolve the average tree depth for a fitted forest, with fallback.
        """
        depths = []
        estimators = getattr(model, "estimators_", None)
        if estimators is not None:
            for est in estimators:
                if hasattr(est, "get_depth"):
                    depths.append(est.get_depth())

        if depths:
            return max(1, int(np.mean(depths)))

        return max(1, int(np.log2(max(n_samples, 2))))

    # ------------------------------------------------------------
    # FLOP estimation
    # ------------------------------------------------------------

    def _estimate_fit_flop(self, X: np.ndarray, y: Any) -> int:
        """
        Estimate training FLOPs for common sklearn models.
        """
        if X.ndim != 2:
            # FLOP formulas assume standard tabular input (n_samples, n_features).
            # For non-2D inputs the analytical cost model is not reliable,
            # so we conservatively return 0.
            return 0

        n_samples, n_features = X.shape
        model = self.model

        # ---------------- LinearRegression ---------------- #
        if isinstance(model, LinearRegression):
            flop = 2 * n_samples * (n_features ** 2) + (2 / 3) * (n_features ** 3)
            return int(flop)

        # ---------------- Ridge / Lasso ---------------- #
        if isinstance(model, (Ridge, Lasso)):
            iters = self._resolve_effective_iterations(model, n_samples)
            flop = iters * n_samples * n_features
            return int(flop)

        # ---------------- LogisticRegression ---------------- #
        if isinstance(model, LogisticRegression):
            iters = self._resolve_effective_iterations(model, n_samples)
            n_classes = len(np.unique(y)) if y is not None else 1
            flop = iters * n_samples * n_features * max(n_classes, 1)
            return int(flop)

        # ---------------- SGDClassifier / SGDRegressor ---------------- #
        if isinstance(model, (SGDClassifier, SGDRegressor)):
            iters = self._resolve_effective_iterations(model, n_samples)

            if y is not None and isinstance(model, SGDClassifier):
                n_classes = len(np.unique(y))
            else:
                n_classes = 1

            # t_ already represents update steps, so we do not multiply by n_samples again
            flop = iters * n_features * max(n_classes, 1)
            return int(flop)

        # ---------------- KNN ---------------- #
        if isinstance(model, (KNeighborsClassifier, KNeighborsRegressor)):
            algorithm = getattr(model, "algorithm", "auto")

            if algorithm == "brute":
                return int(n_samples * n_features)

            if algorithm in ("kd_tree", "ball_tree"):
                return int(n_samples * n_features * np.log2(max(n_samples, 2)))

            return int(n_samples * n_features)

        # ---------------- Decision Tree ---------------- #
        if isinstance(model, (DecisionTreeClassifier, DecisionTreeRegressor)):
            flop = n_samples * n_features * np.log2(max(n_samples, 2))
            return int(flop)

        # ---------------- Random Forest ---------------- #
        if isinstance(model, (RandomForestClassifier, RandomForestRegressor)):
            n_trees = getattr(model, "n_estimators", 100)
            flop = n_trees * n_samples * n_features * np.log2(max(n_samples, 2))
            return int(flop)

        # ---------------- Linear SVM ---------------- #
        if isinstance(model, (LinearSVC, LinearSVR)):
            iters = self._resolve_effective_iterations(model, n_samples)
            flop = iters * n_samples * n_features
            return int(flop)

        # ---------------- Kernel SVM ---------------- #
        if isinstance(model, (SVC, SVR)):
            flop = (n_samples ** 2) * n_features
            return int(flop)

        # ---------------- KMeans ---------------- #
        if isinstance(model, KMeans):
            k = getattr(model, "n_clusters", 8)
            iters = self._resolve_effective_iterations(model, n_samples)
            flop = iters * n_samples * n_features * k
            return int(flop)

        raise Exception(f"FLOPs estimation not implemented for {model.__class__.__name__} model type!")

    def _estimate_predict_flop(self, X: np.ndarray, y: np.ndarray) -> int:
        if X.ndim != 2:
            # Predict formulas assume standard tabular input (n_samples, n_features).
            return 0

        n_samples, n_features = X.shape
        n_outputs = 1 if y.ndim == 1 else y.shape[1]
        model = self.model

        # ---------------- Linear / Logistic / SGD ---------------- #
        if isinstance(
            model,
            (
                LinearRegression,
                Ridge,
                Lasso,
                LogisticRegression,
                SGDClassifier,
                SGDRegressor,
                LinearSVC,
                LinearSVR,
            ),
        ):
            flop = 2 * n_features * n_outputs * n_samples
            return int(flop)

        # ---------------- KNN ---------------- #
        if isinstance(model, (KNeighborsClassifier, KNeighborsRegressor)):
            n_train = getattr(model, "n_samples_fit_", None)
            if n_train is None and hasattr(model, "_fit_X"):
                n_train = model._fit_X.shape[0]
            if n_train is None:
                return 0

            flop = 2 * int(n_train) * n_features * n_samples
            return int(flop)

        # ---------------- Decision Tree predict ---------------- #
        if isinstance(model, (DecisionTreeClassifier, DecisionTreeRegressor)):
            depth = self._resolve_tree_depth(model, n_samples)
            flop = n_samples * depth
            return int(flop)

        # ---------------- Random Forest predict ---------------- #
        if isinstance(model, (RandomForestClassifier, RandomForestRegressor)):
            n_trees = getattr(model, "n_estimators", 100)
            avg_depth = self._resolve_forest_avg_depth(model, n_samples)
            flop = n_trees * n_samples * avg_depth
            return int(flop)

        # ---------------- Kernel SVM ---------------- #
        if isinstance(model, (SVC, SVR)):
            support_vectors = getattr(model, "support_vectors_", None)
            n_sv = support_vectors.shape[0] if support_vectors is not None else n_samples
            flop = 2 * n_sv * n_features * n_samples
            return int(flop)

        # ---------------- KMeans predict ---------------- #
        if isinstance(model, KMeans):
            k = getattr(model, "n_clusters", 8)
            flop = 2 * n_samples * n_features * k
            return int(flop)

        raise Exception(f"FLOPs estimation not implemented for {model.__class__.__name__} model type!")

    def _estimate_transform_flop(self, X: np.ndarray, Z: np.ndarray) -> int:
        """
        Estimate FLOPs for common sklearn transform methods.
        """
        if X.ndim != 2:
            # Transform formulas assume standard tabular input (n_samples, n_features).
            return 0

        n_samples, n_features = X.shape
        model = self.model

        # ---------------- StandardScaler ---------------- #
        if isinstance(model, StandardScaler):
            return int(2 * n_samples * n_features)

        # ---------------- MinMaxScaler ---------------- #
        if isinstance(model, MinMaxScaler):
            return int(2 * n_samples * n_features)

        # ---------------- Normalizer ---------------- #
        if isinstance(model, Normalizer):
            return int(4 * n_samples * n_features)

        # ---------------- PCA ---------------- #
        if isinstance(model, PCA):
            n_components = getattr(model, "n_components_", None)
            if n_components is None:
                n_components = getattr(model, "n_components", None)
            if n_components is None:
                n_components = Z.shape[1] if Z.ndim == 2 else n_features

            return int(2 * n_samples * n_features * int(n_components))

        # ---------------- KMeans.transform ---------------- #
        if isinstance(model, KMeans):
            k = getattr(model, "n_clusters", 8)
            return int(2 * n_samples * n_features * k)

        raise Exception(f"FLOPs estimation not implemented for {model.__class__.__name__} model type!")
