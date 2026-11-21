"""Real-coded Genetic Algorithm (tournament selection, BLX-alpha crossover,
Gaussian mutation) over a box-bounded continuous design space.

Same vectorized `objective` contract as pso.py: it is called with an
(pop_size, dim) array and returns (pop_size,) scores.
"""
from __future__ import annotations

import numpy as np


def genetic_algorithm(objective, bounds, pop_size=60, n_gen=120,
                      cx_rate=0.9, mut_rate=0.12, alpha=0.5, tourney=3,
                      seed=0, minimize=True):
    rng = np.random.default_rng(seed)
    lb = np.asarray([b[0] for b in bounds], float)
    ub = np.asarray([b[1] for b in bounds], float)
    dim = len(bounds)
    span = ub - lb
    sign = 1.0 if minimize else -1.0

    def f(P):
        return sign * np.asarray(objective(P), float).ravel()

    pop = rng.uniform(lb, ub, size=(pop_size, dim))
    fit = f(pop)
    bi = int(np.argmin(fit))
    best, best_f = pop[bi].copy(), float(fit[bi])

    history = [sign * best_f]
    n_evals = pop_size
    for _ in range(n_gen):
        # --- tournament selection ---
        idx = rng.integers(0, pop_size, size=(pop_size, tourney))
        winners = idx[np.arange(pop_size), np.argmin(fit[idx], axis=1)]
        parents = pop[winners].copy()
        rng.shuffle(parents)

        # --- BLX-alpha crossover ---
        children = parents.copy()
        for i in range(0, pop_size - 1, 2):
            if rng.random() < cx_rate:
                a, b = parents[i], parents[i + 1]
                lo = np.minimum(a, b) - alpha * np.abs(a - b)
                hi = np.maximum(a, b) + alpha * np.abs(a - b)
                children[i] = np.clip(rng.uniform(lo, hi), lb, ub)
                children[i + 1] = np.clip(rng.uniform(lo, hi), lb, ub)

        # --- Gaussian mutation ---
        mask = rng.random((pop_size, dim)) < mut_rate
        noise = rng.normal(0.0, 0.1, size=(pop_size, dim)) * span
        pop = np.clip(children + mask * noise, lb, ub)

        fit = f(pop)
        n_evals += pop_size
        gi = int(np.argmin(fit))
        if fit[gi] < best_f:
            best_f = float(fit[gi])
            best = pop[gi].copy()
        history.append(sign * best_f)

    return {
        "x": best,
        "fx": sign * best_f,
        "history": np.asarray(history),
        "n_evals": n_evals,
    }
