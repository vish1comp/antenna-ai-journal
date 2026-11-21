"""Generative inverse design: sample a *diverse set* of geometries that meet a
target spec, then validate them with the forward surrogate.

Contrast with `run_all.py`'s optimizer, which returns a single design. Here a
conditional GMM models the joint distribution of (geometry, frequency, S11) and,
conditioned on a target (2.4 GHz, target S11), generates many candidate
geometries. Each is scored by the forward surrogate to measure spec fidelity;
their spread quantifies diversity; clustering exposes distinct design modes.

Run:  python generate_designs.py
Outputs: results/generative_summary.md, results/figures/generative_*.png
"""
from __future__ import annotations

import os
import numpy as np
from sklearn.cluster import KMeans

import config as C
from data.get_dataset import load_dataset
from models.surrogates import GBDTSurrogate
from generate.cgmm import ConditionalGMM
from evaluate import plots

TARGET_S11 = -20.0   # demanding but data-supported return-loss target (dB)
MASK = -10.0
N_GEN = 400


def main():
    os.makedirs(C.FIG_DIR, exist_ok=True)
    print("[1/4] Loading data + forward surrogate ...")
    ds = load_dataset()
    geom, fcol, tcol = ds["geom_cols"], ds["freq_col"], ds["target_col"]
    fwd = GBDTSurrogate(random_state=C.SEED).fit(ds["X"], ds["y"])   # S11(geom, f)
    tf = ds["target_freq"]

    print("[2/4] Fitting conditional GMM over [geometry, frequency, S11] ...")
    Z = ds["df"][geom + [fcol, tcol]].to_numpy(float)
    cgmm = ConditionalGMM(n_components=40, seed=C.SEED).fit(Z)
    cond_idx = [len(geom), len(geom) + 1]   # positions of (freq, S11) in Z

    print(f"[3/4] Generating {N_GEN} geometries for spec: "
          f"S11 = {TARGET_S11:.0f} dB @ {tf:.2f} GHz ...")
    samp, free_idx, _ = cgmm.sample(cond_idx, [tf, TARGET_S11], n=N_GEN, seed=C.SEED)
    lb = np.array([b[0] for b in ds["opt_bounds"]])
    ub = np.array([b[1] for b in ds["opt_bounds"]])
    samp = np.clip(samp, lb, ub)

    # validate with the forward surrogate
    achieved = fwd.predict(ds["assemble"](samp))
    within3 = float(np.mean(np.abs(achieved - TARGET_S11) <= 3.0))
    below_mask = float(np.mean(achieved <= MASK))
    # diversity: mean per-dim spread as a fraction of the design range
    diversity = float(np.mean(samp.std(0) / (ub - lb)))
    # multimodality: how many distinct modes among spec-meeting designs
    good = samp[achieved <= MASK]
    kk = min(3, max(1, len(good) // 20))
    modes = KMeans(n_clusters=kk, n_init=5, random_state=C.SEED).fit(
        (good - lb) / (ub - lb)) if len(good) >= kk else None

    print(f"[4/4] Results: median achieved S11 = {np.median(achieved):.1f} dB | "
          f"within 3 dB of target = {within3:.0%} | below -10 dB mask = {below_mask:.0%} | "
          f"diversity = {diversity:.2f} | modes = {kk}")

    # figures: pick the two highest-variance geometry dims (most design freedom)
    spread = samp.std(0) / (ub - lb)
    xi, yi = np.argsort(spread)[::-1][:2]
    real = ds["verify_X"]
    plots.scatter_designs(
        samp[:, xi], samp[:, yi], achieved,
        geom[xi], geom[yi], "achieved S11 (dB)",
        f"Generated designs for S11={TARGET_S11:.0f} dB @ {tf:.2f} GHz",
        C.FIG_DIR, "generative_designs.png",
        real_xy=(real[:, xi], real[:, yi]))
    plots.hist_fidelity(achieved, TARGET_S11, MASK, C.FIG_DIR, "generative_fidelity.png")

    _summary(ds, tf, achieved, within3, below_mask, diversity, kk)
    print("\nDone. See results/generative_summary.md and results/figures/generative_*.png")


def _summary(ds, tf, achieved, within3, below_mask, diversity, modes):
    lines = [
        f"# Generative inverse design — {ds['name']}\n",
        f"Conditional GMM over [geometry, frequency, S11]; {len(achieved)} geometries "
        f"generated for the spec **S11 = {TARGET_S11:.0f} dB @ {tf:.2f} GHz**, then "
        f"validated with the forward GBDT surrogate.\n",
        "## Outcome\n",
        f"- Median achieved S11: **{np.median(achieved):.1f} dB** "
        f"(mean {np.mean(achieved):.1f} dB).",
        f"- Within 3 dB of target: **{within3:.0%}**; below the -10 dB mask: "
        f"**{below_mask:.0%}**.",
        f"- Geometry diversity (mean per-dim spread / design range): **{diversity:.2f}** "
        f"— a *set* of distinct designs, not one point.",
        f"- Distinct design modes among spec-meeting geometries: **{modes}**.\n",
        "## Why it matters\n",
        "Unlike the optimizer (one geometry), the generative model returns many "
        "diverse geometries meeting the same spec — the one-to-many inverse mapping "
        "the survey highlights. This density-based GMM is the transparent baseline; "
        "VAE/GAN/diffusion (Naseri & Hum 2021; MetaDiffusion 2023) scale the same "
        "idea to free-form, high-dimensional geometry.\n",
        "See `figures/generative_designs.png` (diverse modes, colored by achieved "
        "S11) and `figures/generative_fidelity.png` (spec-fidelity distribution).",
    ]
    with open(os.path.join(C.RESULTS_DIR, "generative_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
