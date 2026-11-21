"""Re-run the ensemble + conformal part of run_uncertainty() (stage 6 of
run_all.py) with the paper's seed and save the arrays to results/uq_data.npz,
so Fig. 4 can be redrawn from the real data.

    python regen_uq_data.py
"""
from __future__ import annotations

import os

import numpy as np
from sklearn.model_selection import train_test_split

import config as C
import run_all


def main():
    ds = run_all.load_dataset()
    X_tr, X_te, y_tr, y_te = train_test_split(ds["X"], ds["y"], test_size=C.TEST_SIZE, random_state=C.SEED)
    Xf, Xc, yf, yc = train_test_split(X_tr, y_tr, test_size=0.25, random_state=C.SEED)
    ens = run_all.EnsembleSurrogate(n_models=C.UQ["n_models"], seed=C.SEED).fit(Xf, yf)
    mu_c, sd_c = ens.predict_mean_std(Xc)
    mu_t, sd_t = ens.predict_mean_std(X_te)
    a = C.UQ["alpha"]
    cf = run_all.split_conformal(yc, mu_c, alpha=a)
    cfn = run_all.split_conformal(yc, mu_c, alpha=a, sigma_cal=sd_c)
    cov = float(np.mean((y_te >= mu_t - cf["q"]) & (y_te <= mu_t + cf["q"])))
    covn = float(np.mean((y_te >= mu_t - cfn["q"] * sd_t) & (y_te <= mu_t + cfn["q"] * sd_t)))
    print(f"n_models={C.UQ['n_models']}  coverage fixed={cov:.3f}  scaled={covn:.3f}  q={cf['q']:.3f}")
    out = os.path.join(C.RESULTS_DIR, "uq_data.npz")
    np.savez(out, y_cal=yc, mu_cal=mu_c, sd_cal=sd_c, y_test=y_te, mu_test=mu_t, sd_test=sd_t,
             cov=cov, covn=covn, alpha=a, n_models=C.UQ["n_models"])
    print("wrote", out)


if __name__ == "__main__":
    main()
