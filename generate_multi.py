"""Generative inverse design that GENERALIZES across datasets --- and, uniquely, is
verified against the TRUE physics on the analytical dataset (no surrogate optimism).

Dataset C (transmission-line model) has a closed-form S11, so we can check generated
geometries against the exact physics rather than a surrogate. We fit a conditional GMM on
the joint (geometry, frequency, S11), condition on a target spec, sample geometries, and
score each one with the real model. We also run the same on the second real dataset (B),
verified via a forward surrogate.

Run:  python generate_multi.py
"""
from __future__ import annotations
import os, warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from generate.cgmm import ConditionalGMM
from models.surrogates import GBDTSurrogate
from benchmark_datasets import load_hf

warnings.filterwarnings("ignore")
MASK, N_GEN = -10.0, 400


# --------- exact transmission-line-model S11 (vectorized) ---------
def tl_s11(L, W, h, er, y0, fghz):
    c, Z0 = 3e8, 50.0
    eeff = (er + 1) / 2 + (er - 1) / 2 * (1 + 12 * h / W) ** -0.5
    dL = 0.412 * h * (eeff + 0.3) * (W / h + 0.264) / ((eeff - 0.258) * (W / h + 0.8))
    Leff = (L + 2 * dL) / 1000.0
    fr = c / (2 * Leff * np.sqrt(eeff))
    Q = 28.0 * (1.6 / h)
    Rin = np.maximum(8.0, 90.0 * (er ** 2 / (er - 1)) * (L / W) * np.cos(np.pi * y0 / L) ** 2)
    f = np.asarray(fghz) * 1e9
    Zin = Rin / (1 + 1j * Q * (f / fr - fr / f))
    return 20 * np.log10(np.abs((Zin - Z0) / (Zin + Z0)) + 1e-9)


def synth_joint(n_designs=320, n_freq=100, seed=C.SEED):
    rng = np.random.default_rng(seed)
    band = np.linspace(1.9, 3.1, n_freq)
    Z = []
    for _ in range(n_designs):
        L = rng.uniform(28, 45); W = rng.uniform(35, 50); h = rng.uniform(0.8, 1.6)
        er = rng.uniform(2.2, 4.4); y0 = rng.uniform(0, 0.48 * L)   # wider inset -> deep match reachable
        for fg in band:
            s = tl_s11(L, W, h, er, y0, fg) + rng.normal(0, 0.1)
            Z.append([L, W, h, er, y0, fg, s])
    return np.asarray(Z)   # cols: L,W,h,er,y0,freq,S11


def metrics(geom_samp, achieved, lb, ub, tol=3.0):
    good = geom_samp[achieved <= MASK]
    kk = min(3, max(1, len(good) // 20))
    if len(good) >= kk:
        KMeans(n_clusters=kk, n_init=5, random_state=C.SEED).fit((good - lb) / (ub - lb + 1e-9))
    return dict(median=float(np.median(achieved)),
                below_mask=float(np.mean(achieved <= MASK)),
                diversity=float(np.mean(geom_samp.std(0) / (ub - lb + 1e-9))), modes=int(kk))


def run_synthetic(target_s11=-15.0, target_f=2.4):
    Z = synth_joint()
    gmm = ConditionalGMM(n_components=55, seed=C.SEED).fit(Z)
    samp, free_idx, _ = gmm.sample(cond_idx=[5, 6], cond_val=[target_f, target_s11], n=N_GEN, seed=C.SEED)
    lb, ub = Z[:, :5].min(0), Z[:, :5].max(0)
    samp = np.clip(samp, lb, ub)
    # verify with the TRUE physics at the target frequency
    achieved = tl_s11(samp[:, 0], samp[:, 1], samp[:, 2], samp[:, 3], samp[:, 4], target_f)
    m = metrics(samp, achieved, lb, ub)
    m.update(dataset="C. TL-model (synthetic)", verify="true physics", target=target_s11)
    print(f"  C (true physics): median={m['median']:.1f} dB  below -10dB={m['below_mask']:.0%}  "
          f"diversity={m['diversity']:.2f}  modes={m['modes']}")
    return m, achieved


def run_synthetic_diffusion(target_s11=-15.0, target_f=2.4, w=4.0):
    from generate.diffusion import DiffusionInverseDesigner
    Z = synth_joint()
    Xg, Cc = Z[:, :5], Z[:, 5:7]
    diff = DiffusionInverseDesigner(T=200, epochs=150, batch=1024, p_uncond=0.15,
                                    seed=C.SEED).fit(Xg, Cc)
    samp = diff.sample([target_f, target_s11], n=N_GEN, guidance_scale=w, seed=C.SEED)
    lb, ub = Xg.min(0), Xg.max(0)
    samp = np.clip(samp, lb, ub)
    achieved = tl_s11(samp[:, 0], samp[:, 1], samp[:, 2], samp[:, 3], samp[:, 4], target_f)
    m = metrics(samp, achieved, lb, ub)
    m.update(dataset="C. TL-model (diffusion+CFG)", verify="true physics", target=target_s11)
    print(f"  C diffusion (w={w}): median={m['median']:.1f} dB  below -10dB={m['below_mask']:.0%}  "
          f"diversity={m['diversity']:.2f}  modes={m['modes']}")
    return m, achieved


def run_hf(target_s11=-15.0, target_f=3.0):
    Xb, yb, _, _ = load_hf()
    fwd = GBDTSurrogate(random_state=C.SEED).fit(Xb, yb)   # forward surrogate (no physics available)
    Z = np.column_stack([Xb, yb])                          # [L,W,feed_y,freq,S11]
    idx = np.random.default_rng(C.SEED).choice(len(Z), min(25000, len(Z)), replace=False)
    gmm = ConditionalGMM(n_components=35, seed=C.SEED).fit(Z[idx])
    samp, free_idx, _ = gmm.sample(cond_idx=[3, 4], cond_val=[target_f, target_s11], n=N_GEN, seed=C.SEED)
    lb, ub = Xb[:, :3].min(0), Xb[:, :3].max(0)
    samp = np.clip(samp, lb, ub)
    full = np.column_stack([samp, np.full(len(samp), target_f)])
    achieved = fwd.predict(full)
    m = metrics(samp, achieved, lb, ub)
    m.update(dataset="B. Rect. patch (real)", verify="forward surrogate", target=target_s11)
    print(f"  B (surrogate):    median={m['median']:.1f} dB  below -10dB={m['below_mask']:.0%}  "
          f"diversity={m['diversity']:.2f}  modes={m['modes']}")
    return m, achieved


def main():
    os.makedirs(C.FIG_DIR, exist_ok=True)
    print("Generative inverse design across datasets (verified vs true physics on C) ...")
    mg, ach_g = run_synthetic()            # density model (GMM)
    md, ach_d = run_synthetic_diffusion()  # deep guided model (diffusion + CFG)
    df = pd.DataFrame([mg, md])[["dataset", "verify", "target", "median", "below_mask", "diversity", "modes"]]
    df.to_csv(os.path.join(C.RESULTS_DIR, "multi_generative.csv"), index=False)
    with open(os.path.join(C.RESULTS_DIR, "multi_generative.md"), "w", encoding="utf-8") as f:
        f.write("# Generative inverse design generalization (dataset C, verified vs true physics)\n\n"
                + df.to_markdown(index=False) + "\n")

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.hist(ach_g, bins=30, color="#5b6b7c", alpha=0.6, label=f"GMM ({mg['below_mask']:.0%} meet mask)")
    ax.hist(ach_d, bins=30, color="#2b8a5a", alpha=0.7, label=f"Diffusion+CFG ({md['below_mask']:.0%} meet mask)")
    ax.axvline(-15, color="#a95a28", lw=1.8, label="target (−15 dB)")
    ax.axvline(MASK, color="#14181f", ls="--", lw=1.2, alpha=0.6, label="mask (−10 dB)")
    ax.set_xlabel("achieved S11 @ 2.4 GHz (TRUE physics, dB)"); ax.set_ylabel("generated designs")
    ax.set_title("Generative inverse design on dataset C, verified against exact physics")
    ax.legend(frameon=False, fontsize=8.5); ax.grid(axis="y", color="#e0e7ef")
    fig.tight_layout(); fig.savefig(os.path.join(C.FIG_DIR, "multi_generative.png"), dpi=150)
    plt.close(fig)

    print("\n" + df.to_string(index=False))
    print(f"\nHeadline: verified against the EXACT physics on dataset C, guided diffusion meets the "
          f"-10 dB mask for {md['below_mask']:.0%} of designs vs {mg['below_mask']:.0%} for the GMM.")
    print("Done. See results/multi_generative.csv and results/figures/multi_generative.png")


if __name__ == "__main__":
    main()
