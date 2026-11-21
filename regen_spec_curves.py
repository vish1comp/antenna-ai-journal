"""Re-run only the spec-matching design (stage 5 of run_all.py) with the paper's
seed and save the curves to results/spec_curves.npz, so Fig. 3 can be redrawn
from the real data.

    python regen_spec_curves.py
"""
from __future__ import annotations

import os

import numpy as np
from sklearn.model_selection import train_test_split

import config as C
import run_all
from evaluate import plots

captured = {}


def _capture(freqs, s11_pred, spec, outdir, name, true_freqs=None, true_s11=None):
    captured.update(freqs=np.asarray(freqs), s11=np.asarray(s11_pred), true_freqs=np.asarray(true_freqs),
                    true_s11=np.asarray(true_s11), band=np.asarray(spec["band"]), mask=spec["mask_db"],
                    center=spec["center"])


def main():
    ds = run_all.load_dataset()
    X_tr, X_te, y_tr, y_te = train_test_split(ds["X"], ds["y"], test_size=C.TEST_SIZE, random_state=C.SEED)
    metrics_df, fitted = run_all.benchmark_surrogates(X_tr, y_tr, X_te, y_te, ds["target_name"])
    best = metrics_df.sort_values("NRMSE").iloc[0]["model"]
    print("best surrogate:", best)
    plots.response_curve = _capture
    run_all.plots.response_curve = _capture
    res = run_all.run_spec_design(fitted[best], ds)
    ev = res["eval"]
    out = os.path.join(C.RESULTS_DIR, "spec_curves.npz")
    np.savez(out, **captured, center_s11=ev["center_s11"], min_s11=ev["min_s11"],
             bandwidth_ghz=ev["bandwidth_ghz"], verify_dist=res["verify_dist"])
    print("wrote", out)


if __name__ == "__main__":
    main()
