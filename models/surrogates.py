"""Surrogate models with a common API for antenna performance prediction.

Every surrogate exposes the same interface so the optimizer and the benchmark
harness can treat them interchangeably:

    s = MLPSurrogate().fit(X_train, y_train)
    y_hat = s.predict(X_test)           # (n,) predictions
    std   = s.predict_std(X_test)       # (n,) 1-sigma, GP only (has_std=True)

Feature/target standardization is handled internally so callers pass raw
design-parameter arrays. These three families mirror the survey's comparison
axis: a neural network (flexible, data-hungry), gradient boosting (strong
tabular baseline), and a Gaussian process (sample-efficient, gives uncertainty).
"""
from __future__ import annotations

import time
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel


class Surrogate:
    """Base class defining the shared fit/predict contract."""

    name = "base"
    has_std = False

    def __init__(self):
        self.model = None
        self.fit_time_ = None

    def fit(self, X, y):
        t = time.perf_counter()
        self.model.fit(np.asarray(X, float), np.asarray(y, float).ravel())
        self.fit_time_ = time.perf_counter() - t
        return self

    def predict(self, X):
        return np.asarray(self.model.predict(np.asarray(X, float))).ravel()

    def predict_std(self, X):  # pragma: no cover - overridden where available
        raise NotImplementedError(f"{self.name} does not provide uncertainty")


class MLPSurrogate(Surrogate):
    """Multilayer perceptron (two hidden layers) with input standardization."""

    name = "MLP"

    def __init__(self, random_state: int = 0):
        super().__init__()
        self.model = Pipeline([
            ("scale", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(128, 128),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=1200,
                early_stopping=True,
                n_iter_no_change=30,
                validation_fraction=0.15,
                random_state=random_state,
            )),
        ])


class GBDTSurrogate(Surrogate):
    """Histogram gradient-boosted trees - fast, strong tabular baseline."""

    name = "GBDT"

    def __init__(self, random_state: int = 0):
        super().__init__()
        self.model = HistGradientBoostingRegressor(
            learning_rate=0.08,
            max_iter=500,
            max_depth=None,
            l2_regularization=1.0,
            early_stopping=True,
            random_state=random_state,
        )


class GPSurrogate(Surrogate):
    """Gaussian process (RBF + noise). Sample-efficient and returns 1-sigma.

    O(n^3) training, so callers should cap the training set (see run_all.py).
    """

    name = "GP"
    has_std = True

    def __init__(self, random_state: int = 0):
        super().__init__()
        kernel = (
            ConstantKernel(1.0, (1e-2, 1e3))
            * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))
            + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e1))
        )
        self._scaler = StandardScaler()
        self._gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=random_state,
        )

    def fit(self, X, y):
        t = time.perf_counter()
        Xs = self._scaler.fit_transform(np.asarray(X, float))
        self._gp.fit(Xs, np.asarray(y, float).ravel())
        self.fit_time_ = time.perf_counter() - t
        return self

    def predict(self, X):
        return np.asarray(self._gp.predict(self._scaler.transform(np.asarray(X, float)))).ravel()

    def predict_std(self, X):
        _, std = self._gp.predict(self._scaler.transform(np.asarray(X, float)), return_std=True)
        return np.asarray(std).ravel()


ALL_SURROGATES = {
    "MLP": MLPSurrogate,
    "GBDT": GBDTSurrogate,
    "GP": GPSurrogate,
}
