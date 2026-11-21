"""Cracking generative transfer with PHYSICS-INFORMED data curation.

Earlier (generate_multi.py) generative inverse design transferred poorly to the synthetic
dataset because the training designs were spread randomly across 1.8-3.2 GHz, so the target
(deep S11 *at 2.4 GHz*) was sparsely represented. Here we curate the training set by the
physics: choose each design's length so its resonance lands near the 2.4 GHz band (inverting
the transmission-line formula), and vary the feed inset for a range of match depths. The
conditional generators then have dense support at the target, and -- verified against the
EXACT physics -- their mask compliance jumps. This is the realistic use case: train the
inverse-design generator on data relevant to the target band.

Run:  python generate_curated.py
"""
from __future__ import annotations
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from generate.cgmm import ConditionalGMM
from generate.diffusion import DiffusionInverseDesigner
from generate_multi import tl_s11, metrics, synth_joint, MASK, N_GEN

warnings.filterwarnings("ignore")
TARGET_S11, TARGET_F = -15.0, 2.4


def solve_L(fr_ghz, W, h, er):
    """Invert the resonance formula: length that resonates at fr_ghz (mm)."""
    c = 3e8
    eeff = (er + 1) / 2 + (er - 1) / 2 * (1 + 12 * h / W) ** -0.5
    Leff = c / (2 * fr_ghz * 1e9 * np.sqrt(eeff)) * 1000.0
    dL = 0.412 * h * (eeff + 0.3) * (W / h + 0.264) / ((eeff - 0.258) * (W / h + 0.8))
    return Leff - 2 * dL


def curated_joint(n_designs=340, n_freq=90, seed=C.SEED):
    rng = np.random.default_rng(seed)
    band = np.linspace(2.15, 2.65, n_freq)
    Z = []
    for _ in range(n_designs):
        W = rng.uniform(35, 50); h = rng.uniform(0.8, 1.6); er = rng.uniform(2.2, 4.4)
        fr = rng.uniform(2.3, 2.5)                       # resonate tightly IN the target band
        L = float(np.clip(solve_L(fr, W, h, er), 26, 48))
        y0 = rng.uniform(0, 0.48 * L)
        for fg in band:
            Z.append([L, W, h, er, y0, fg, tl_s11(L, W, h, er, y0, fg) + rng.normal(0, 0.1)])
    return np.asarray(Z)


def gen_gmm(Z, tag):
    gmm = ConditionalGMM(n_components=55, seed=C.SEED).fit(Z)
    samp, _, _ = gmm.sample([5, 6], [TARGET_F, TARGET_S11], n=N_GEN, seed=C.SEED)
    lb, ub = Z[:, :5].min(0), Z[:, :5].max(0); samp = np.clip(samp, lb, ub)
    ach = tl_s11(samp[:, 0], samp[:, 1], samp[:, 2], samp[:, 3], samp[:, 4], TARGET_F)
    m = metrics(samp, ach, lb, ub); m.update(method="GMM", data=tag)
    print(f"  {tag:9s} GMM       median={m['median']:5.1f} dB  mask={m['below_mask']:.0%}")
    return m, ach


def gen_diff(Z, tag, w=4.0):
    diff = DiffusionInverseDesigner(T=200, epochs=180, batch=1024, p_uncond=0.15, seed=C.SEED)
    diff.fit(Z[:, :5], Z[:, 5:7])
    samp = diff.sample([TARGET_F, TARGET_S11], n=N_GEN, guidance_scale=w, seed=C.SEED)
    lb, ub = Z[:, :5].min(0), Z[:, :5].max(0); samp = np.clip(samp, lb, ub)
    ach = tl_s11(samp[:, 0], samp[:, 1], samp[:, 2], samp[:, 3], samp[:, 4], TARGET_F)
    m = metrics(samp, ach, lb, ub); m.update(method="Diffusion+CFG", data=tag)
    print(f"  {tag:9s} Diffusion median={m['median']:5.1f} dB  mask={m['below_mask']:.0%}")
    return m, ach


def main():
    os.makedirs(C.FIG_DIR, exist_ok=True)
    print("Physics-informed data curation for generative transfer (verified vs true physics):")
    Zb, Zc = synth_joint(), curated_joint()
    rows, ach = [], {}
    for Z, tag in [(Zb, "broad"), (Zc, "curated")]:
        mg, ag = gen_gmm(Z, tag); rows.append(mg); ach[(tag, "GMM")] = ag
        md, ad = gen_diff(Z, tag); rows.append(md); ach[(tag, "Diffusion+CFG")] = ad

    df = pd.DataFrame(rows)[["data", "method", "median", "below_mask", "diversity", "modes"]]
    df["below_mask"] = (df["below_mask"] * 100).round(1)
    df.to_csv(os.path.join(C.RESULTS_DIR, "generative_curated.csv"), index=False)
    with open(os.path.join(C.RESULTS_DIR, "generative_curated.md"), "w", encoding="utf-8") as f:
        f.write("# Physics-informed curation for generative transfer (true-physics verified)\n\n"
                + df.to_markdown(index=False) + "\n")

    # grouped bar: mask compliance, broad vs curated x GMM vs diffusion
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    labels = ["GMM", "Diffusion+CFG"]
    broad = [df[(df.data == "broad") & (df.method == m)]["below_mask"].values[0] for m in labels]
    cur = [df[(df.data == "curated") & (df.method == m)]["below_mask"].values[0] for m in labels]
    x = np.arange(len(labels)); w = 0.36
    ax.bar(x - w/2, broad, w, label="broad training", color="#5b6b7c")
    ax.bar(x + w/2, cur, w, label="physics-curated training", color="#2b8a5a")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("% meeting −10 dB mask (true physics)")
    ax.set_ylim(0, 105); ax.set_title("Physics-informed curation cracks generative transfer")
    ax.legend(frameon=False); ax.grid(axis="y", color="#e0e7ef")
    for xi, b, c in zip(x, broad, cur):
        ax.text(xi - w/2, b + 1, f"{b:.0f}", ha="center", fontsize=9)
        ax.text(xi + w/2, c + 1, f"{c:.0f}", ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(C.FIG_DIR, "generative_curated.png"), dpi=150)
    plt.close(fig)

    print("\n" + df.to_string(index=False))
    print("Done. See results/generative_curated.csv and results/figures/generative_curated.png")


if __name__ == "__main__":
    main()
