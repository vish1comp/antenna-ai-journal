"""Uncertainty quantification for surrogates.

Two complementary tools the survey argues are prerequisites for trusting an
AI-proposed antenna design:

* `EnsembleSurrogate` - a bootstrap committee giving a predictive **mean** and an
  **epistemic std** (disagreement), used both as a reliability signal and to make
  optimization risk-averse.
* `split_conformal` - distribution-free calibrated prediction intervals (a half
  width with a finite-sample coverage guarantee), with an optional locally
  adaptive variant normalized by the ensemble std.
"""
from __future__ import annotations

import numpy as np
from models.surrogates import GBDTSurrogate


class EnsembleSurrogate:
    """Bootstrap GBDT ensemble -> predictive mean and epistemic std."""

    name = "GBDT ensemble"

    def __init__(self, n_models=6, seed=0):
        self.n_models = n_models
        self.seed = seed
        self.models = []

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float).ravel()
        rng = np.random.default_rng(self.seed)
        self.models = []
        for b in range(self.n_models):
            idx = rng.integers(0, len(X), len(X))          # bootstrap resample
            self.models.append(GBDTSurrogate(random_state=self.seed + b).fit(X[idx], y[idx]))
        return self

    def _stack(self, X):
        return np.vstack([m.predict(X) for m in self.models])   # (n_models, n)

    def predict(self, X):
        return self._stack(X).mean(axis=0)

    def predict_std(self, X):
        return self._stack(X).std(axis=0)

    def predict_mean_std(self, X):
        S = self._stack(X)
        return S.mean(axis=0), S.std(axis=0)


def split_conformal(y_cal, mu_cal, alpha=0.1, sigma_cal=None):
    """Split-conformal half-width at miscoverage `alpha`.

    Standard (sigma_cal=None): constant half-width q, interval [mu-q, mu+q].
    Normalized (sigma_cal given): adaptive half-width q*sigma(x), tighter where
    the model is confident.
    """
    res = np.abs(np.asarray(y_cal, float) - np.asarray(mu_cal, float))
    n = len(res)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    if sigma_cal is None:
        return {"q": float(np.quantile(res, level, method="higher")), "normalized": False}
    s = np.maximum(np.asarray(sigma_cal, float), 1e-9)
    return {"q": float(np.quantile(res / s, level, method="higher")), "normalized": True}
