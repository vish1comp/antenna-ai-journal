"""Publication-style figures for the surrogate + optimization demo.

Uses the non-interactive Agg backend so it runs headless. Every figure is
saved to results/figures/ as PNG at 150 dpi.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ACCENT = "#0b6e78"
COPPER = "#a95a28"
INK = "#0e1620"
PALETTE = ["#0b6e78", "#a95a28", "#2b8a5a", "#5b6b7c", "#8a5aa9"]

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.edgecolor": "#c3cdd8",
    "axes.grid": True,
    "grid.color": "#e0e7ef",
    "grid.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def parity_plot(y_true, y_pred, title, outdir, name, target_name="target"):
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.scatter(y_true, y_pred, s=12, alpha=0.5, color=ACCENT, edgecolor="none")
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], "--", color=COPPER, lw=1.3, label="ideal")
    ax.set_xlabel(f"true {target_name}")
    ax.set_ylabel(f"predicted {target_name}")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    return _save(fig, outdir, name)


def comparison_bars(names, values, ylabel, title, outdir, name, fmt="{:.3f}"):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(names))]
    bars = ax.bar(names, values, color=colors, width=0.62)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                fmt.format(v), ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.margins(y=0.15)
    return _save(fig, outdir, name)


def convergence(histories, labels, ylabel, title, outdir, name):
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for i, (h, lab) in enumerate(zip(histories, labels)):
        ax.plot(np.arange(len(h)), h, color=PALETTE[i % len(PALETTE)], lw=1.8, label=lab)
    ax.set_xlabel("iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    return _save(fig, outdir, name)


def scatter_designs(x, y, c, xlabel, ylabel, clabel, title, outdir, name, real_xy=None):
    """Generated designs in a 2-D geometry slice, colored by achieved metric."""
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    if real_xy is not None:
        ax.scatter(real_xy[0], real_xy[1], s=55, facecolors="none",
                   edgecolors="#5b6b7c", linewidths=1.2, label="real designs")
    sc = ax.scatter(x, y, c=c, s=22, cmap="viridis_r", alpha=0.85, edgecolor="none",
                    label="generated")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(clabel)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8.5, loc="best")
    return _save(fig, outdir, name)


def hist_fidelity(achieved, target, mask, outdir, name, xlabel="achieved S11 @ 2.4 GHz (dB)"):
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.hist(achieved, bins=30, color=ACCENT, alpha=0.75, edgecolor="white", linewidth=0.4)
    ax.axvline(target, color=COPPER, lw=1.8, ls="-", label=f"target ({target:.0f} dB)")
    ax.axvline(mask, color=INK, lw=1.2, ls="--", alpha=0.6, label=f"mask ({mask:.0f} dB)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("generated designs")
    ax.set_title("Generative inverse design: spec fidelity")
    ax.legend(frameon=False, fontsize=8.5)
    return _save(fig, outdir, name)


def guidance_curve(ws, below_mask, diversity, outdir, name):
    """Twin-axis: spec fidelity (up) and diversity (down) vs guidance scale w."""
    fig, ax1 = plt.subplots(figsize=(5.4, 3.9))
    l1 = ax1.plot(ws, np.array(below_mask) * 100, "-o", color=ACCENT, lw=2, ms=4,
                  label="spec fidelity (% below −10 dB)")
    ax1.set_xlabel("guidance scale  w")
    ax1.set_ylabel("% below −10 dB mask", color=ACCENT)
    ax1.tick_params(axis="y", labelcolor=ACCENT)
    ax2 = ax1.twinx()
    ax2.grid(False)
    l2 = ax2.plot(ws, diversity, "-s", color=COPPER, lw=2, ms=4, label="diversity")
    ax2.set_ylabel("design diversity", color=COPPER)
    ax2.tick_params(axis="y", labelcolor=COPPER)
    ax1.set_title("Classifier-free guidance: fidelity vs diversity")
    lines = l1 + l2
    ax1.legend(lines, [ln.get_label() for ln in lines], frameon=False, fontsize=8.5,
               loc="center right")
    return _save(fig, outdir, name)


def hist_compare(series, labels, target, mask, outdir, name,
                 xlabel="achieved S11 @ 2.4 GHz (dB)", title="Generative inverse design"):
    fig, ax = plt.subplots(figsize=(5.9, 3.9))
    for i, (s, lab) in enumerate(zip(series, labels)):
        ax.hist(s, bins=30, alpha=0.55, color=PALETTE[i % len(PALETTE)], label=lab,
                edgecolor="white", linewidth=0.3)
    ax.axvline(target, color=COPPER, lw=1.8, label=f"target ({target:.0f} dB)")
    ax.axvline(mask, color=INK, lw=1.2, ls="--", alpha=0.6, label=f"mask ({mask:.0f} dB)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("generated designs")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8.5)
    return _save(fig, outdir, name)


def reliability(sigma, abs_err, outdir, name, n_bins=10):
    """Does predicted uncertainty track actual error? Bin by sigma, plot RMSE."""
    sigma = np.asarray(sigma, float)
    abs_err = np.asarray(abs_err, float)
    order = np.argsort(sigma)
    bins = np.array_split(order, n_bins)
    xs = [float(sigma[b].mean()) for b in bins]
    ys = [float(np.sqrt((abs_err[b] ** 2).mean())) for b in bins]
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.scatter(sigma, abs_err, s=6, alpha=0.12, color="#5b6b7c", edgecolor="none")
    ax.plot(xs, ys, "-o", color=ACCENT, lw=2, ms=4.5, label="binned RMSE")
    lim = max(max(xs), max(ys)) * 1.05
    ax.plot([0, lim], [0, lim], "--", color=COPPER, lw=1.1, label="ideal (σ = error)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, None)
    ax.set_xlabel("ensemble predictive std σ (dB)")
    ax.set_ylabel("|prediction error| (dB)")
    ax.set_title("Uncertainty reliability")
    ax.legend(frameon=False, fontsize=8.5)
    return _save(fig, outdir, name)


def response_curve(freqs, s11_pred, spec, outdir, name, true_freqs=None, true_s11=None):
    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    ax.axvspan(spec["band"][0], spec["band"][1], color=ACCENT, alpha=0.08,
               label="target band")
    ax.axhline(spec["mask_db"], color=COPPER, ls="--", lw=1.3,
               label=f"mask ({spec['mask_db']:.0f} dB)")
    if true_freqs is not None:
        ax.plot(true_freqs, true_s11, color="#5b6b7c", lw=1.4, alpha=0.9,
                label="nearest real design (true)")
    ax.plot(freqs, s11_pred, "-", lw=2.0, color=ACCENT, label="designed (surrogate)")
    ax.axvline(spec["center"], color=INK, lw=0.8, alpha=0.3)
    ax.set_xlim(spec["plot_range"])
    ax.set_xlabel("frequency (GHz)")
    ax.set_ylabel("S11 (dB)")
    ax.set_title("Inverse spec-matching design")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    return _save(fig, outdir, name)


def convergence_evals(evals_list, hist_list, labels, ylabel, title, outdir, name):
    """Best-so-far vs cumulative objective evaluations (log x) - shows the
    sample-efficiency gap between BO and population methods."""
    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    marks = ["o", "s", "^", "D"]
    for i, (ev, h, lab) in enumerate(zip(evals_list, hist_list, labels)):
        ax.plot(ev, h, marker=marks[i % len(marks)], ms=3, markevery=max(1, len(ev)//12),
                color=PALETTE[i % len(PALETTE)], lw=1.8, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("objective evaluations (log)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    return _save(fig, outdir, name)


def learning_curve(sizes, curves, labels, ylabel, title, outdir, name):
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for i, (c, lab) in enumerate(zip(curves, labels)):
        c = np.asarray(c)
        ax.plot(sizes, c, "-o", ms=3.5, color=PALETTE[i % len(PALETTE)], lw=1.7, label=lab)
    ax.set_xlabel("labeled training samples")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    return _save(fig, outdir, name)
