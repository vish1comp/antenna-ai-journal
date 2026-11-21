"""Cross-dataset generalization study (journal-grade): do the surrogates hold up across
DIFFERENT patch-antenna datasets, and how stable is that across random splits?

Three datasets, same task (geometry + frequency -> S11):
  A. Zenodo CST patch  (real, 55k rows)          -- the primary dataset.
  B. HF rectangular patch (real, public parquet) -- an independent real dataset.
  C. Transmission-line model patch (synthetic)   -- physics-based, fully reproducible.

Four surrogates (MLP, GBDT, Random Forest, GP) are each evaluated over the repeated
random train/test splits listed in SEEDS; we report mean +/- standard deviation of test
R2 and NRMSE.
Consistent accuracy across independent datasets AND across resamples is the evidence that
the methodology generalizes rather than fitting one dataset's quirks or one lucky split.

Run:  python benchmark_datasets.py     (needs pyarrow for dataset B)
"""
from __future__ import annotations
import os, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from data.get_dataset import load_dataset
from models.surrogates import MLPSurrogate, GBDTSurrogate, GPSurrogate
from evaluate.metrics import regression_report

warnings.filterwarnings("ignore")
SEEDS = [42, 7, 123]               # repeated resamples (mean +/- std)
N_MAX = 6000                       # cap rows per dataset for a comparable, tractable study
MODELS = ["MLP", "GBDT", "Random Forest", "GP"]
COLORS = {"MLP": "#0b6e78", "GBDT": "#a95a28", "Random Forest": "#2b8a5a", "GP": "#5b6b7c"}
HF_URL = "https://huggingface.co/datasets/becklabash/rectangular-patch-antenna-freq-response/resolve/main/data/train-00000-of-00001.parquet"
HF_CACHE = os.path.join(C.BASE_DIR, "data", "raw", "rect_patch.parquet")


def make(name, seed):
    if name == "MLP":
        return MLPSurrogate(random_state=seed)
    if name == "GBDT":
        return GBDTSurrogate(random_state=seed)
    if name == "Random Forest":
        return RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
    return GPSurrogate(random_state=seed)


def _subsample(X, y, n, seed=C.SEED):
    if len(X) <= n:
        return X, y
    idx = np.random.default_rng(seed).choice(len(X), n, replace=False)
    return X[idx], y[idx]


# ---------------------------- dataset loaders ----------------------------
def load_zenodo():
    ds = load_dataset()
    return ds["X"], ds["y"], "A. Zenodo CST patch (real)"


def load_hf():
    if os.path.exists(HF_CACHE):
        df = pd.read_parquet(HF_CACHE)
    else:
        df = pd.read_parquet(HF_URL); df.to_parquet(HF_CACHE)
    F = np.stack(df["frequencies"].values) / 1e9
    S = np.stack(df["s11"].values)
    band = np.where((F[0] >= 2.0) & (F[0] <= 4.0))[0]
    F, S = F[:, band], S[:, band]
    geom = np.stack([df["length"].values, df["width"].values, df["feed_y"].values], axis=1).astype(float)
    X = np.hstack([np.repeat(geom, len(band), axis=0), F.reshape(-1, 1)])
    return X, S.reshape(-1).astype(float), "B. HF rectangular patch (real)"


def gen_synthetic(n_designs=90, n_freq=120, seed=C.SEED):
    rng = np.random.default_rng(seed); c = 3e8; Z0 = 50.0
    X, Y = [], []
    for _ in range(n_designs):
        L = rng.uniform(28, 45); W = rng.uniform(35, 50)
        h = rng.uniform(0.8, 1.6); er = rng.uniform(2.2, 4.4); y0 = rng.uniform(0, 0.35 * L)
        eeff = (er + 1) / 2 + (er - 1) / 2 * (1 + 12 * h / W) ** -0.5
        dL = 0.412 * h * (eeff + 0.3) * (W / h + 0.264) / ((eeff - 0.258) * (W / h + 0.8))
        Leff = (L + 2 * dL) / 1000.0
        fr = c / (2 * Leff * np.sqrt(eeff))
        Q = 28.0 * (1.6 / h)
        Rin = max(8.0, 90.0 * (er ** 2 / (er - 1)) * (L / W) * np.cos(np.pi * y0 / L) ** 2)
        for fg in np.linspace(fr * 0.85, fr * 1.15, n_freq) / 1e9:
            f = fg * 1e9
            Zin = Rin / (1 + 1j * Q * (f / fr - fr / f))
            s11 = 20 * np.log10(np.abs((Zin - Z0) / (Zin + Z0)) + 1e-9)
            X.append([L, W, h, er, y0, fg]); Y.append(s11 + rng.normal(0, 0.1))
    return np.asarray(X), np.asarray(Y), "C. TL-model patch (synthetic)"


# ---------------------------- experiment ----------------------------
def benchmark(X, y, name):
    X, y = _subsample(X, y, N_MAX)
    rows = []
    for seed in SEEDS:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed)
        for mname in MODELS:
            m = make(mname, seed)
            if mname == "GP" and len(Xtr) > C.GP_MAX_TRAIN:
                idx = np.random.default_rng(seed).choice(len(Xtr), C.GP_MAX_TRAIN, replace=False)
                m.fit(Xtr[idx], ytr[idx])
            else:
                m.fit(Xtr, ytr)
            rep = regression_report(yte, m.predict(Xte))
            rows.append({"Dataset": name, "Model": mname, "seed": seed, "n": len(X),
                         "d": X.shape[1], **rep})
    return rows


def main():
    os.makedirs(C.FIG_DIR, exist_ok=True)
    raw = []
    for loader in (load_zenodo, load_hf, gen_synthetic):
        print(f"[loading] {loader.__name__} ...")
        try:
            X, y, name = loader()
            print(f"  {name}: {X.shape[0]} rows, {X.shape[1]} features "
                  f"-> {len(SEEDS)} splits x {len(MODELS)} models")
            raw += benchmark(X, y, name)
        except Exception as e:  # noqa: BLE001
            print(f"  SKIPPED ({e})")

    raw = pd.DataFrame(raw)
    raw.to_csv(os.path.join(C.RESULTS_DIR, "cross_dataset_raw.csv"), index=False)
    agg = (raw.groupby(["Dataset", "Model"])
           .agg(R2_mean=("R2", "mean"), R2_std=("R2", "std"),
                NRMSE_mean=("NRMSE", "mean"), NRMSE_std=("NRMSE", "std"),
                n=("n", "first"), d=("d", "first"))
           .reset_index())
    agg["R2"] = agg.apply(lambda r: f"{r.R2_mean:.3f} ± {r.R2_std:.3f}", axis=1)
    agg["NRMSE"] = agg.apply(lambda r: f"{r.NRMSE_mean:.3f} ± {r.NRMSE_std:.3f}", axis=1)
    out = agg[["Dataset", "Model", "R2", "NRMSE", "n", "d"]]
    out.to_csv(os.path.join(C.RESULTS_DIR, "cross_dataset.csv"), index=False)
    with open(os.path.join(C.RESULTS_DIR, "cross_dataset.md"), "w", encoding="utf-8") as f:
        f.write(f"# Cross-dataset generalization ({len(SEEDS)} repeated splits; mean ± std)\n\n"
                + out.to_markdown(index=False) + "\n")

    # grouped bars with std error bars
    piv_m = agg.pivot(index="Dataset", columns="Model", values="R2_mean")[MODELS]
    piv_s = agg.pivot(index="Dataset", columns="Model", values="R2_std")[MODELS]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    xs = np.arange(len(piv_m)); w = 0.2
    for i, mdl in enumerate(MODELS):
        ax.bar(xs + (i - 1.5) * w, piv_m[mdl], w, yerr=piv_s[mdl], capsize=3,
               label=mdl, color=COLORS[mdl], error_kw=dict(lw=1, ecolor="#333"))
    ax.set_xticks(xs); ax.set_xticklabels([d.split(".")[0] + "." for d in piv_m.index])
    ax.set_ylabel(f"R² (test, mean ± std over {len(SEEDS)} splits)"); ax.set_ylim(0, 1.03)
    ax.set_title("Surrogate accuracy across three patch-antenna datasets")
    ax.legend(title="surrogate", frameon=False, ncol=2, fontsize=9)
    ax.grid(axis="y", color="#e0e7ef")
    fig.tight_layout(); fig.savefig(os.path.join(C.FIG_DIR, "cross_dataset.png"), dpi=150)
    plt.close(fig)

    print("\n" + out.to_string(index=False))
    print("\nDone. See results/cross_dataset.csv (mean ± std) and results/figures/cross_dataset.png")


if __name__ == "__main__":
    main()
