"""Bayesian optimization (GP surrogate + Expected Improvement).

The sample-efficient optimizer: it builds a probabilistic model of the objective
and spends evaluations where improvement is most likely, reaching good designs in
~tens of objective evaluations versus the thousands PSO/GA use. The acquisition is
maximized by scoring a large random candidate pool each step - robust and
dependency-light (no inner gradient optimizer).

Same call contract as pso.py / ga.py:
    objective(P) : (n, dim) -> (n,)   ;   returns {x, fx, history, n_evals}
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel


def bayesian_optimization(objective, bounds, n_init=12, n_iter=60, xi=0.01,
                          n_candidates=4000, seed=0, minimize=True):
    rng = np.random.default_rng(seed)
    lb = np.asarray([b[0] for b in bounds], float)
    ub = np.asarray([b[1] for b in bounds], float)
    dim = len(bounds)
    sign = 1.0 if minimize else -1.0

    def f(P):
        return sign * np.asarray(objective(P), float).ravel()

    X = rng.uniform(lb, ub, size=(n_init, dim))
    y = f(X)
    bi = int(np.argmin(y))
    best_x, best_y = X[bi].copy(), float(y[bi])
    history = [sign * best_y]
    n_evals = n_init

    for _ in range(n_iter):
        scaler = StandardScaler().fit(X)
        kernel = (ConstantKernel(1.0, (1e-2, 1e3))
                  * RBF(1.0, (1e-1, 1e2))
                  + WhiteKernel(1e-3, (1e-6, 1e0)))
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                      n_restarts_optimizer=1, random_state=seed)
        gp.fit(scaler.transform(X), y)

        cand = rng.uniform(lb, ub, size=(n_candidates, dim))
        mu, sd = gp.predict(scaler.transform(cand), return_std=True)
        sd = np.maximum(sd, 1e-9)
        imp = best_y - mu - xi                     # improvement over incumbent (min)
        z = imp / sd
        ei = imp * norm.cdf(z) + sd * norm.pdf(z)  # Expected Improvement
        x_next = cand[int(np.argmax(ei))]

        y_next = float(f(x_next[None, :])[0])
        X = np.vstack([X, x_next])
        y = np.append(y, y_next)
        n_evals += 1
        if y_next < best_y:
            best_y, best_x = y_next, x_next.copy()
        history.append(sign * best_y)

    return {"x": best_x, "fx": sign * best_y,
            "history": np.asarray(history), "n_evals": n_evals}
