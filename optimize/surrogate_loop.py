"""Surrogate-assisted optimization and active learning.

Two ideas from the survey made concrete and measurable:

1. `active_learning` compares sampling strategies at equal labeling budgets:
     * "random"      - passive baseline
     * "uncertainty" - add points a GP is least sure about (naive max-variance)
     * "qbc"         - query-by-committee disagreement + batch diversity
                       (uncertainty AND spread, the standard fix for the
                       redundant clustering that hurts naive max-variance on a
                       structured frequency-sweep pool)

2. `surrogate_objective` turns a trained surrogate into the cheap objective a
   metaheuristic (PSO/GA) optimizes; because there is no live EM solver, the
   proposed optimum is verified against the dataset with `nearest_true`.
"""
from __future__ import annotations

import numpy as np

from models.surrogates import GPSurrogate, GBDTSurrogate
from evaluate.metrics import nrmse


def surrogate_objective(surrogate):
    """Return a vectorized f(P)->(n,) that the optimizers can call."""
    def f(P):
        P = np.atleast_2d(np.asarray(P, float))
        return surrogate.predict(P)
    return f


def nearest_true(x, X_data, y_data):
    """Nearest real design to `x` and its measured target (verification proxy)."""
    x = np.asarray(x, float).ravel()
    d = np.linalg.norm(np.asarray(X_data, float) - x, axis=1)
    j = int(np.argmin(d))
    return {"y_true": float(np.asarray(y_data, float).ravel()[j]),
            "distance": float(d[j]), "index": j}


def _committee_disagreement(X_lab, y_lab, X_query, n_models=5, seed=0):
    """Std of a bootstrap GBDT committee's predictions on X_query (epistemic proxy)."""
    preds = []
    for b in range(n_models):
        rng = np.random.default_rng(seed + b)
        bs = rng.choice(len(X_lab), size=len(X_lab), replace=True)
        m = GBDTSurrogate(random_state=seed + b).fit(X_lab[bs], y_lab[bs])
        preds.append(m.predict(X_query))
    return np.std(np.vstack(preds), axis=0)


def _diverse_batch(cand_idx, X, need, feat_scale):
    """Farthest-first selection of `need` points from candidates already ranked
    by uncertainty (cand_idx[0] = most uncertain). Adds spatial spread so the
    batch is not a cluster of near-duplicate high-variance points."""
    cand = np.asarray(cand_idx)
    if need >= len(cand):
        return [int(c) for c in cand]
    Xc = X[cand] / feat_scale
    chosen = [0]                                   # seed with the most uncertain
    d = np.linalg.norm(Xc - Xc[0], axis=1)
    for _ in range(need - 1):
        nxt = int(np.argmax(d))
        chosen.append(nxt)
        d = np.minimum(d, np.linalg.norm(Xc - Xc[nxt], axis=1))
    return [int(cand[c]) for c in chosen]


def active_learning(pool_X, pool_y, test_X, test_y, sizes, *,
                    strategy="random", init=30, seed=0):
    """Grow a labeled set to each budget in `sizes`; return test NRMSE per size."""
    rng = np.random.default_rng(seed)
    pool_X = np.asarray(pool_X, float)
    pool_y = np.asarray(pool_y, float).ravel()
    n = len(pool_X)
    feat_scale = np.where(pool_X.std(axis=0) > 0, pool_X.std(axis=0), 1.0)

    labeled = list(rng.choice(n, size=min(init, n), replace=False))
    remaining = [i for i in range(n) if i not in set(labeled)]

    errors = []
    for target_size in sizes:
        while len(labeled) < target_size and remaining:
            need = target_size - len(labeled)
            rem = np.asarray(remaining)

            if strategy == "uncertainty" and len(labeled) >= 8:
                gp = GPSurrogate(random_state=seed).fit(pool_X[labeled], pool_y[labeled])
                score = gp.predict_std(pool_X[rem])
                pick = [remaining[k] for k in np.argsort(-score)[:need]]

            elif strategy == "qbc" and len(labeled) >= 8:
                dis = _committee_disagreement(pool_X[labeled], pool_y[labeled],
                                              pool_X[rem], seed=seed)
                top = np.argsort(-dis)[: min(len(rem), max(need * 8, need))]
                cand = rem[top]                      # global idx, ranked by disagreement
                pick = _diverse_batch(cand, pool_X, need, feat_scale)

            else:  # random (and cold-start for the others)
                pick = list(rng.choice(remaining, size=min(need, len(remaining)),
                                       replace=False))

            labeled.extend(pick)
            picks = set(pick)
            remaining = [i for i in remaining if i not in picks]

        model = GBDTSurrogate(random_state=seed).fit(pool_X[labeled], pool_y[labeled])
        errors.append(nrmse(test_y, model.predict(test_X)))

    return errors
