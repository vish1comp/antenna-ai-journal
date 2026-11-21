"""Visualize the microstrip patch-antenna geometry from the dataset parameters.

Draws a to-scale top view of a representative design (the deepest-S11 / best-matched
design at 2.4 GHz) using the actual dataset dimensions, next to that design's measured
S11-vs-frequency response. For showing "what the antenna looks like" and which dataset
was used.

Run:  python visualize_geometry.py   ->  results/figures/antenna_geometry.png
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

import config as C
from data.get_dataset import load_dataset

FR4, COPPER, EDGE, INK, NAVY = "#cfe3cf", "#c8842e", "#8a5a1e", "#14181f", "#0B3D66"


def dim(ax, x0, y0, x1, y1, label, off=0, ha="center", va="center", color=NAVY):
    ax.annotate("", (x1, y1), (x0, y0),
                arrowprops=dict(arrowstyle="<->", color=color, lw=1))
    ax.text((x0+x1)/2 + (off if x0 == x1 else 0), (y0+y1)/2 + (0 if x0 == x1 else off),
            label, color=color, fontsize=8.5, ha=ha, va=va,
            rotation=90 if x0 == x1 else 0)


def main():
    os.makedirs(C.FIG_DIR, exist_ok=True)
    ds = load_dataset()
    df, geom = ds["df"], ds["geom_cols"]
    fcol, tcol, tf = ds["freq_col"], ds["target_col"], ds["target_freq"]

    # representative design: deepest S11 near 2.4 GHz
    near = df[np.isclose(df[fcol], tf, atol=0.01)]
    row = near.loc[near[tcol].idxmin()]
    g = {c: float(row[c]) for c in geom}
    m = np.all([np.isclose(df[c], g[c]) for c in geom], axis=0)
    sweep = df[m].sort_values(fcol)

    Ws = g["width substrate in mm"]; Ls = g["substrate length in mm"]
    Wp = g["patch width in mm"];     Lp = g["patch legth in mm"]
    Lf = g["length impedance line in mm"]; Wf = 3.0
    sw = g["width slot in mm"];       sl = g["length slot in mm"]

    fig, (axg, axs) = plt.subplots(1, 2, figsize=(11.5, 5.2), gridspec_kw={"width_ratios": [1, 1.05]})

    # ---- geometry (top view), substrate centered at origin ----
    axg.add_patch(Rectangle((-Ws/2, -Ls/2), Ws, Ls, facecolor=FR4, edgecolor="#7fae7f", lw=1.2))
    # patch centered
    axg.add_patch(Rectangle((-Wp/2, -Lp/2), Wp, Lp, facecolor=COPPER, edgecolor=EDGE, lw=1.2))
    # microstrip feed line (from substrate bottom edge up to patch bottom edge)
    feed_top = -Lp/2
    axg.add_patch(Rectangle((-Wf/2, feed_top - Lf), Wf, Lf, facecolor=COPPER, edgecolor=EDGE, lw=1))
    # slot etched in the patch (near the radiating edge)
    axg.add_patch(Rectangle((-sl/2, Lp/2 - sw - 1.5), sl, sw, facecolor=FR4, edgecolor=EDGE, lw=0.8))

    pad = 6
    axg.set_xlim(-Ws/2 - pad, Ws/2 + pad); axg.set_ylim(-Ls/2 - Lf - pad, Ls/2 + pad)
    axg.set_aspect("equal"); axg.axis("off")
    axg.set_title("Microstrip patch geometry (top view, to scale)", color=NAVY, fontsize=12)

    dim(axg, -Ws/2, Ls/2 + 2.5, Ws/2, Ls/2 + 2.5, f"$W_s$ = {Ws:.0f} mm", off=1.2, va="bottom")
    dim(axg, Ws/2 + 2.5, -Ls/2, Ws/2 + 2.5, Ls/2, f"$L_s$ = {Ls:.0f} mm", off=1.2)
    dim(axg, -Wp/2, 0, Wp/2, 0, f"$W_p$ = {Wp:.0f}", off=1.0, va="bottom", color=INK)
    dim(axg, -Wp/2 - 1, -Lp/2, -Wp/2 - 1, Lp/2, f"$L_p$ = {Lp:.1f}", off=-1.4, color=INK)
    axg.text(0, feed_top - Lf/2, f"feed\n$L_f$={Lf:.0f}", ha="center", va="center", fontsize=8, color="white")
    axg.text(0, Lp/2 - sw - 1.5 + sw/2, f"slot {sl:.0f}×{sw:.0f}", ha="center", va="center", fontsize=7.5, color=INK)
    axg.text(-Ws/2, -Ls/2 - Lf - pad + 1.5, "FR-4 substrate, h = 1.6 mm; copper 0.035 mm",
             fontsize=8, color="#5a6470")

    # ---- S11 response of this design ----
    axs.plot(sweep[fcol], sweep[tcol], color="#1C6FB8", lw=2)
    axs.axhline(-10, color="#B5771F", ls="--", lw=1.3, label="-10 dB mask")
    axs.axvline(tf, color="#586472", lw=0.8, alpha=0.5)
    smin = sweep[tcol].min()
    axs.plot(tf, float(near.loc[near[tcol].idxmin(), tcol]), "o", color=NAVY, ms=6)
    axs.set_xlabel("frequency (GHz)"); axs.set_ylabel("S11 (dB)")
    axs.set_title(f"Measured response (this design): S11 = {smin:.1f} dB", color=NAVY, fontsize=12)
    axs.grid(True, color="#e6edf4"); axs.legend(frameon=False, fontsize=9)

    fig.suptitle("Dataset: Simulated Patch Antenna for 2.4 GHz (Zenodo 17527112, CC-BY) — CST full-wave",
                 color=NAVY, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(C.FIG_DIR, "antenna_geometry.png")
    fig.savefig(out, dpi=160); plt.close(fig)
    print("Representative design dimensions (mm):")
    for k, v in g.items():
        print(f"  {k:28s} {v:.2f}")
    print(f"  -> S11 at {tf:.2f} GHz = {float(near.loc[near[tcol].idxmin(), tcol]):.2f} dB")
    print("saved", out)


if __name__ == "__main__":
    main()
