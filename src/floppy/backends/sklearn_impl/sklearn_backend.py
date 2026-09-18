from __future__ import annotations
from typing import Any, Callable, Optional
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression, SGDClassifier, SGDRegressor
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.decomposition import PCA
from ..base import BaseBackend
from ...utils.utility import cprint, Color

import numpy as np


class SklearnBackend(BaseBackend):
    """
    Backend for scikit-learn models.

    Responsibilities:
    - Wraps sklearn methods (fit, predict, predict_proba, transform)
    - Estimates model FLOPs using algorithmic complexity heuristics
    - Maps 'fit' to Backward (Training) and 'predict/transform' to Forward (Inference)
    """

    def __init__(self, model, logger=None):
        super().__init__(model, logger=logger)
        self._orig_fit: Optional[Callable] = None
        self._orig_predict: Optional[Callable] = None
        self._orig_predict_proba: Optional[Callable] = None
        self._orig_transform: Optional[Callable] = None
        self.total_forward_memory_bytes: int = 0
        self.total_backward_memory_bytes: int = 0

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
    # Wrapped methods (with Fail-Safe Try/Except)
    # ------------------------------------------------------------

    def _wrap_fit(self, fn: Callable) -> Callable:
        def wrapped(X, y=None, *args, **kwargs):
            result = fn(X, y, *args, **kwargs)
            try:
                x_arr = np.asarray(X)
                flop = self._estimate_fit_flop(x_arr, y)
                bit_width = self._get_numpy_bit_width(x_arr)
                bops = int(flop * bit_width)
                mem_bytes = x_arr.nbytes
                if y is not None:
                    mem_bytes += np.asarray(y).nbytes

                self._accumulate_call(flop, bops, memory_bytes=mem_bytes, is_training=True)

            except Exception as e:
                cprint(f"FLOPpy Warning: Failed to estimate fit FLOPs - {e}", Color.WARNING)

            return result

        return wrapped

    def _wrap_predict(self, fn: Callable) -> Callable:
        def wrapped(X, *args, **kwargs):
            y_pred = fn(X, *args, **kwargs)
            x_arr = np.asarray(X)
            flop = self._estimate_predict_flop(x_arr, np.asarray(y_pred))
            bit_width = self._get_numpy_bit_width(x_arr)
            bops = int(flop * bit_width)
            mem_bytes = x_arr.nbytes + np.asarray(y_pred).nbytes
            self._accumulate_call(flop, bops, memory_bytes=mem_bytes, is_training=False)
            return y_pred

        return wrapped

    def _wrap_predict_proba(self, fn: Callable) -> Callable:
        def wrapped(X, *args, **kwargs):
            proba = fn(X, *args, **kwargs)
            x_arr = np.asarray(X)
            flop = self._estimate_predict_flop(x_arr, np.asarray(proba))
            bit_width = self._get_numpy_bit_width(x_arr)
            bops = int(flop * bit_width)
            mem_bytes = x_arr.nbytes + np.asarray(proba).nbytes
            self._accumulate_call(flop, bops, memory_bytes=mem_bytes, is_training=False)
            return proba

        return wrapped

    def _wrap_transform(self, fn: Callable) -> Callable:
        def wrapped(X, *args, **kwargs):
            z = fn(X, *args, **kwargs)
            x_arr = np.asarray(X)
            flop = self._estimate_transform_flop(x_arr, np.asarray(z))
            bit_width = self._get_numpy_bit_width(x_arr)
            bops = int(flop * bit_width)
            mem_bytes = x_arr.nbytes + np.asarray(z).nbytes
            self._accumulate_call(flop, bops, memory_bytes=mem_bytes, is_training=False)
            return z

        return wrapped

    # ------------------------------------------------------------
    # FLOP accumulation
    # ------------------------------------------------------------

    def _accumulate_call(self, flop: int, bop: int, memory_bytes: int = 0, is_training: bool = False):
        self._batch_idx += 1
        value_flop = int(flop)
        value_bop = int(bop)
        value_mem = int(memory_bytes)

        if is_training:
            self._last_batch_backward_flop = value_flop
            self._last_batch_backward_bop = value_bop
            self.total_backward_flop += value_flop
            self.total_backward_bop += value_bop
            self.total_backward_memory_bytes += value_mem

        else:
            self._last_batch_flop = value_flop
            self._last_batch_bop = value_bop
            self.total_forward_flop += value_flop
            self.total_forward_bop += value_bop
            self.total_forward_memory_bytes += value_mem

    def _get_numpy_bit_width(self, x_arr: np.ndarray) -> int:
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

        return 64

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

    def _estimate_fit_flop(self, X: np.ndarray, y: Any, current_model=None) -> int:
        """
        Estimate training FLOPs for common sklearn models.
        """
        if current_model is None:
            current_model = self.model

        if X.ndim != 2:
            return 0

        n_samples, n_features = X.shape

        # ---------------- Meta Estimator: MultiOutput ---------------- #
        if isinstance(current_model, (MultiOutputClassifier, MultiOutputRegressor)):
            if hasattr(current_model, "estimators_"):
                total_flops = 0
                for est in current_model.estimators_:
                    total_flops += self._estimate_fit_flop(X, y, current_model=est)

                return total_flops

            return 0

        # ---------------- XGBoost / LightGBM / CatBoost ---------------- #
        model_name = current_model.__class__.__name__
        if "XGB" in model_name or "LGBM" in model_name or "CatBoost" in model_name:
            n_trees = getattr(current_model, "n_estimators", 100)
            depth = getattr(current_model, "max_depth", 6)
            if depth is None or depth <= 0:
                depth = 6

            flop = n_trees * n_samples * n_features * depth * np.log2(max(n_samples, 2))
            return int(flop)

        # ---------------- LinearRegression ---------------- #
        if isinstance(current_model, LinearRegression):
            flop = 2 * n_samples * (n_features ** 2) + (2 / 3) * (n_features ** 3)
            return int(flop)

        # ---------------- Ridge / Lasso ---------------- #
        if isinstance(current_model, (Ridge, Lasso)):
            iters = self._resolve_effective_iterations(current_model, n_samples)
            flop = iters * n_samples * n_features
            return int(flop)

        # ---------------- LogisticRegression ---------------- #
        if isinstance(current_model, LogisticRegression):
            iters = self._resolve_effective_iterations(current_model, n_samples)
            n_classes = len(np.unique(y)) if y is not None else 1
            flop = iters * n_samples * n_features * max(n_classes, 1)
            return int(flop)

        # ---------------- SGDClassifier / SGDRegressor ---------------- #
        if isinstance(current_model, (SGDClassifier, SGDRegressor)):
            iters = self._resolve_effective_iterations(current_model, n_samples)
            if y is not None and isinstance(current_model, SGDClassifier):
                n_classes = len(np.unique(y))

            else:
                n_classes = 1

            flop = iters * n_features * max(n_classes, 1)
            return int(flop)

        # ---------------- KNN ---------------- #
        if isinstance(current_model, (KNeighborsClassifier, KNeighborsRegressor)):
            algorithm = getattr(current_model, "algorithm", "auto")
            if algorithm == "brute":
                return int(n_samples * n_features)

            if algorithm in ("kd_tree", "ball_tree"):
                return int(n_samples * n_features * np.log2(max(n_samples, 2)))

            return int(n_samples * n_features)

        # ---------------- Decision Tree ---------------- #
        if isinstance(current_model, (DecisionTreeClassifier, DecisionTreeRegressor)):
            flop = n_samples * n_features * np.log2(max(n_samples, 2))
            return int(flop)

        # ---------------- Random Forest ---------------- #
        if isinstance(current_model, (RandomForestClassifier, RandomForestRegressor)):
            n_trees = getattr(current_model, "n_estimators", 100)
            flop = n_trees * n_samples * n_features * np.log2(max(n_samples, 2))
            return int(flop)

        # ---------------- Linear SVM ---------------- #
        if isinstance(current_model, (LinearSVC, LinearSVR)):
            iters = self._resolve_effective_iterations(current_model, n_samples)
            flop = iters * n_samples * n_features
            return int(flop)

        # ---------------- Kernel SVM ---------------- #
        if isinstance(current_model, (SVC, SVR)):
            flop = (n_samples ** 2) * n_features
            return int(flop)

        # ---------------- KMeans ---------------- #
        if isinstance(current_model, KMeans):
            k = getattr(current_model, "n_clusters", 8)
            iters = self._resolve_effective_iterations(current_model, n_samples)
            flop = iters * n_samples * n_features * k
            return int(flop)

        raise Exception(f"FLOPs estimation not implemented for {current_model.__class__.__name__} model type!")

    def _estimate_predict_flop(self, X: np.ndarray, y: np.ndarray, current_model=None) -> int:
        if current_model is None:
            current_model = self.model

        if X.ndim != 2:
            return 0

        n_samples, n_features = X.shape
        n_outputs = 1 if y.ndim == 1 else y.shape[1]

        # ---------------- Meta Estimator: MultiOutput ---------------- #
        if isinstance(current_model, (MultiOutputClassifier, MultiOutputRegressor)):
            if hasattr(current_model, "estimators_"):
                total_flops = 0
                for est in current_model.estimators_:
                    total_flops += self._estimate_predict_flop(X, y, current_model=est)

                return total_flops
            return 0

        # ---------------- XGBoost / LightGBM / CatBoost ---------------- #
        model_name = current_model.__class__.__name__
        if "XGB" in model_name or "LGBM" in model_name or "CatBoost" in model_name:
            n_trees = getattr(current_model, "n_estimators", 100)
            depth = getattr(current_model, "max_depth", 6)
            if depth is None or depth <= 0:
                depth = self._resolve_tree_depth(current_model, n_samples)

            flop = n_trees * n_samples * depth
            return int(flop)

        # ---------------- Linear / Logistic / SGD ---------------- #
        if isinstance(current_model, (LinearRegression, Ridge, Lasso, LogisticRegression, SGDClassifier, SGDRegressor, LinearSVC, LinearSVR)):
            flop = 2 * n_features * n_outputs * n_samples
            return int(flop)

        # ---------------- KNN ---------------- #
        if isinstance(current_model, (KNeighborsClassifier, KNeighborsRegressor)):
            n_train = getattr(current_model, "n_samples_fit_", None)
            if n_train is None and hasattr(current_model, "_fit_X"):
                n_train = current_model._fit_X.shape[0]

            if n_train is None:
                return 0

            flop = 2 * int(n_train) * n_features * n_samples
            return int(flop)

        # ---------------- Decision Tree predict ---------------- #
        if isinstance(current_model, (DecisionTreeClassifier, DecisionTreeRegressor)):
            depth = self._resolve_tree_depth(current_model, n_samples)
            flop = n_samples * depth
            return int(flop)

        # ---------------- Random Forest predict ---------------- #
        if isinstance(current_model, (RandomForestClassifier, RandomForestRegressor)):
            n_trees = getattr(current_model, "n_estimators", 100)
            avg_depth = self._resolve_forest_avg_depth(current_model, n_samples)
            flop = n_trees * n_samples * avg_depth
            return int(flop)

        # ---------------- Kernel SVM ---------------- #
        if isinstance(current_model, (SVC, SVR)):
            support_vectors = getattr(current_model, "support_vectors_", None)
            n_sv = support_vectors.shape[0] if support_vectors is not None else n_samples
            flop = 2 * n_sv * n_features * n_samples
            return int(flop)

        # ---------------- KMeans predict ---------------- #
        if isinstance(current_model, KMeans):
            k = getattr(current_model, "n_clusters", 8)
            flop = 2 * n_samples * n_features * k
            return int(flop)

        raise Exception(f"FLOPs estimation not implemented for {current_model.__class__.__name__} model type!")

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
