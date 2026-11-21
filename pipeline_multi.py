"""Whole-framework generalization: run the FULL pipeline on every dataset.

For each of three independent patch-antenna datasets we run the complete chain used in the
paper --- train a surrogate, optimize a geometry with PSO, verify the proposed design
against the nearest real measurement, and calibrate uncertainty with a deep ensemble plus
conformal prediction --- and collect one consolidated table. Consistent behavior across
datasets is the evidence that the framework (not just one result) generalizes.

Run:  python pipeline_multi.py
"""
from __future__ import annotations
import os, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from data.get_dataset import load_dataset
from models.surrogates import MLPSurrogate
from models.uncertainty import EnsembleSurrogate, split_conformal
from optimize.pso import particle_swarm
from evaluate.metrics import regression_report
from benchmark_datasets import load_hf, gen_synthetic, _subsample, N_MAX

warnings.filterwarnings("ignore")


def descriptors():
    ds = load_dataset()
    A = dict(name="A. Zenodo CST (real)", X=ds["X"], y=ds["y"],
             geom_idx=list(range(7)), freq_idx=7, target=2.4)
    Xb, yb, _, _ = load_hf()
    B = dict(name="B. Rect. patch (real)", X=Xb, y=yb,
             geom_idx=[0, 1, 2], freq_idx=3, target=3.0)
    Xc, yc, _, _ = gen_synthetic()
    Cd = dict(name="C. TL-model (synth.)", X=Xc, y=yc,
              geom_idx=[0, 1, 2, 3, 4], freq_idx=5, target=2.4)
    return [A, B, Cd]


def run_one(d):
    X, y = _subsample(np.asarray(d["X"], float), np.asarray(d["y"], float), N_MAX)
    gi, fi, tf = d["geom_idx"], d["freq_idx"], d["target"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=C.SEED)

    # 1) surrogate
    mlp = MLPSurrogate(random_state=C.SEED).fit(Xtr, ytr)
    r2 = regression_report(yte, mlp.predict(Xte))["R2"]

    # 2) optimize a geometry to minimize S11 at the target band
    lb, ub = X[:, gi].min(0), X[:, gi].max(0)
    freqs = X[:, fi]
    nf = float(freqs[np.argmin(np.abs(freqs - tf))])

    def assemble(P):
        P = np.atleast_2d(np.asarray(P, float))
        full = np.zeros((len(P), X.shape[1]))
        full[:, gi] = P
        full[:, fi] = nf
        return full

    res = particle_swarm(lambda P: mlp.predict(assemble(P)),
                         list(zip(lb, ub)), minimize=True, seed=C.SEED,
                         n_particles=40, n_iters=100)
    opt_s11 = float(res["fx"])

    # 3) verify against nearest real design near the target band
    band = np.abs(X[:, fi] - tf) <= 0.1
    Xr, yr = X[band][:, gi], y[band]
    j = int(np.argmin(np.linalg.norm(Xr - res["x"], axis=1)))
    real_s11 = float(yr[j])

    # 4) uncertainty: ensemble + conformal coverage
    Xf, Xc, yf, yc = train_test_split(Xtr, ytr, test_size=0.25, random_state=C.SEED)
    ens = EnsembleSurrogate(n_models=6, seed=C.SEED).fit(Xf, yf)
    cf = split_conformal(yc, ens.predict(Xc), alpha=0.10)
    mu_t = ens.predict(Xte)
    cov = float(np.mean((yte >= mu_t - cf["q"]) & (yte <= mu_t + cf["q"])))

    print(f"  {d['name']:24s} R2={r2:.3f}  opt S11={opt_s11:6.1f} dB  "
          f"real={real_s11:6.1f} dB  cov(90%)={cov:.3f}  @~{nf:.2f} GHz")
    return {"Dataset": d["name"], "MLP R2": round(r2, 3),
            "Opt S11 (dB)": round(opt_s11, 1), "Nearest real (dB)": round(real_s11, 1),
            "Conformal cov. (target 90%)": round(cov * 100, 1), "Band (GHz)": round(nf, 2)}


def main():
    os.makedirs(C.FIG_DIR, exist_ok=True)
    print("Running the full framework on each dataset ...")
    rows = [run_one(d) for d in descriptors()]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.RESULTS_DIR, "multi_dataset.csv"), index=False)
    with open(os.path.join(C.RESULTS_DIR, "multi_dataset.md"), "w", encoding="utf-8") as f:
        f.write("# Whole-framework generalization across datasets\n\n"
                + df.to_markdown(index=False) + "\n")

    # figure: conformal coverage per dataset (calibration generalizes)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    labels = [r["Dataset"].split(".")[0] + "." for r in rows]
    cov = [r["Conformal cov. (target 90%)"] for r in rows]
    r2 = [r["MLP R2"] * 100 for r in rows]
    x = np.arange(len(labels)); w = 0.36
    ax.bar(x - w/2, r2, w, label="MLP R² (%)", color="#0b6e78")
    ax.bar(x + w/2, cov, w, label="Conformal coverage (%)", color="#a95a28")
    ax.axhline(90, ls="--", lw=1.2, color="#586472", label="90% target")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 105)
    ax.set_ylabel("%"); ax.set_title("Framework generalization across datasets")
    ax.legend(frameon=False, fontsize=9); ax.grid(axis="y", color="#e0e7ef")
    for xi, a, b in zip(x, r2, cov):
        ax.text(xi - w/2, a + 1, f"{a:.0f}", ha="center", fontsize=8)
        ax.text(xi + w/2, b + 1, f"{b:.0f}", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(C.FIG_DIR, "multi_dataset.png"), dpi=150)
    plt.close(fig)

    print("\n" + df.to_string(index=False))
    print("\nDone. See results/multi_dataset.csv and results/figures/multi_dataset.png")


if __name__ == "__main__":
    main()
