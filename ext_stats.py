"""Statistics for the improved-paper experiments (results/improved/).

    python ext_stats.py          -> prints summaries and Holm-corrected paired tests

Paired protocol as in ms_stats.py: Shapiro-Wilk on the paired differences selects a paired
t-test or Wilcoxon; Holm correction within each family.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
from ms_stats import paired, holm, stars   # noqa: E402

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "improved")
OURS = "GBDT ensemble (ours)"


def fam(title, rows):
    print("\n==", title)
    adj = holm([r["p"] for r in rows])
    for r, q in zip(rows, adj):
        r["p_holm"] = q
        print("  %-38s %-9s d=%+8.3f  p=%.2e  pH=%.2e %s" % (r["name"], r["test"], r["d"], r["p"], q, stars(q)))
    return rows


def uq():
    d = pd.read_csv(os.path.join(R, "uq_baselines.csv"))
    g = d.groupby(["dataset", "method"])
    s = g[["cov_raw", "cov_conf", "width_conf", "rmse", "spearman"]].agg(["mean", "std"])
    s[("fit_s", "median")] = g["fit_s"].median()
    print(s.round(3).to_string())
    s.to_csv(os.path.join(R, "uq_baselines_summary.csv"))
    out = []
    for ds in "ABC":
        x = d[d.dataset == ds].pivot(index="seed", columns="method")
        rows = []
        for m in x["width_conf"].columns:
            if m == OURS:
                continue
            a, b = x["width_conf"][m].values, x["width_conf"][OURS].values
            t, dd, p = paired(a, b)
            rows.append(dict(dataset=ds, name=m, test=t, d=dd, p=p))
        out += fam("conformal width, %s: baseline minus ours" % ds, rows)
        # coverage error after calibration: all should be ~0
        rows = []
        for m in x["cov_conf"].columns:
            rows.append(dict(dataset=ds, name=m + " |cov-0.9|", test="-", d=float(np.mean(np.abs(x["cov_conf"][m] - 0.9))), p=1.0))
        print("  calibrated coverage error:", {r["name"]: round(r["d"], 4) for r in rows})
    pd.DataFrame(out).to_csv(os.path.join(R, "uq_baselines_tests.csv"), index=False)


def inv():
    d = pd.read_csv(os.path.join(R, "inv_baselines.csv"))
    d["secs_per_ok"] = d["secs"] / np.maximum(d["below_mask"] * d["n_designs"], 1e-9)
    s = d.groupby(["data", "method"])[["n_designs", "median", "hit3", "below_mask", "diversity",
                                       "secs", "secs_per_design", "secs_per_ok", "surrogate_evals"]].agg(["mean", "std"])
    print(s.round(3).to_string())
    s.to_csv(os.path.join(R, "inv_baselines_summary.csv"))
    out = []
    for data in ("broad", "curated"):
        x = d[d.data == data].pivot(index="seed", columns="method")
        for metric in ("below_mask", "secs_per_ok"):
            rows = []
            for m in ("Multi-start PSO", "GMM", "Tandem", "Random"):
                a, b = x[metric]["Diffusion+CFG"].values, x[metric][m].values
                t, dd, p = paired(a, b)
                rows.append(dict(data=data, metric=metric, name="Diffusion - " + m, test=t, d=dd, p=p))
            out += fam("%s %s" % (data, metric), rows)
    pd.DataFrame(out).to_csv(os.path.join(R, "inv_baselines_tests.csv"), index=False)


def bsize():
    p = os.path.join(R, "b_learning_curve.csv")
    d = pd.read_csv(p)
    s = d.groupby(["model", "n_train"])[["R2", "NRMSE", "fit_s"]].agg(["mean", "std", "count"])
    print(s.round(3).to_string())
    s.to_csv(os.path.join(R, "b_learning_curve_summary.csv"))


if __name__ == "__main__":
    for f in sys.argv[1:] or ["uq", "inv", "bsize"]:
        globals()[f]()
