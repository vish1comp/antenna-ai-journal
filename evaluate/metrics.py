"""Regression metrics used across the surrogate benchmark.

NRMSE is normalized by the target range so it is comparable across datasets
and targets with different units (the survey's tables mix accuracy%, RMSE and
iteration counts with no common yardstick - this fixes that).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def nrmse(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    rng = float(np.ptp(y_true))
    return rmse(y_true, y_pred) / rng if rng > 0 else float("nan")


def regression_report(y_true, y_pred):
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "NRMSE": nrmse(y_true, y_pred),
    }
