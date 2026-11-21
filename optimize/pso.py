"""Particle Swarm Optimization over a box-bounded continuous design space.

Dependency-free (numpy only) and fully vectorized: `objective` receives an
(n_particles, dim) array and returns an (n_particles,) array of scores, so a
surrogate model can score the whole swarm in one batched `.predict` call.
"""
from __future__ import annotations

import numpy as np


def particle_swarm(objective, bounds, n_particles=40, n_iters=120,
                   w=0.72, c1=1.49, c2=1.49, seed=0, minimize=True):
    rng = np.random.default_rng(seed)
    lb = np.asarray([b[0] for b in bounds], float)
    ub = np.asarray([b[1] for b in bounds], float)
    dim = len(bounds)
    span = ub - lb
    sign = 1.0 if minimize else -1.0  # internal problem is always a minimization

    def f(P):
        return sign * np.asarray(objective(P), float).ravel()

    X = rng.uniform(lb, ub, size=(n_particles, dim))
    V = rng.uniform(-span, span, size=(n_particles, dim)) * 0.1

    pbest = X.copy()
    pbest_val = f(X)
    gi = int(np.argmin(pbest_val))
    gbest = pbest[gi].copy()
    gbest_val = float(pbest_val[gi])

    history = [sign * gbest_val]
    n_evals = n_particles
    for _ in range(n_iters):
        r1 = rng.random((n_particles, dim))
        r2 = rng.random((n_particles, dim))
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        X = np.clip(X + V, lb, ub)

        val = f(X)
        n_evals += n_particles
        improved = val < pbest_val
        pbest[improved] = X[improved]
        pbest_val[improved] = val[improved]

        gi = int(np.argmin(pbest_val))
        if pbest_val[gi] < gbest_val:
            gbest_val = float(pbest_val[gi])
            gbest = pbest[gi].copy()
        history.append(sign * gbest_val)

    return {
        "x": gbest,
        "fx": sign * gbest_val,
        "history": np.asarray(history),
        "n_evals": n_evals,
    }
