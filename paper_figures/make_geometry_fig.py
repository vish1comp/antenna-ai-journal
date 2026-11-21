"""Fig. 2 of the journal extension: the antenna designed and validated in the
MAPCON paper [2], redrawn to scale in the same visual style as the MAPCON
figure (green substrate, orange patch, slot near the top edge, inset feed).

Dimensions are those printed in the MAPCON paper: Lp = 29, Ls = 41, Ws = 41,
Wp = 40, h = 1.6, Sw = 6.8, SL = 1 (mm). The feed width and length are not
given numerically there and are drawn schematically (3 mm x 6 mm).

    python make_geometry_fig.py
"""
from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures", "geometry.png")

SUB, PATCH, EDGE_S, EDGE_P, SLOT, INK = "#6CC28F", "#EDBE62", "#3E9A66", "#C9973A", "#7CCBA0", "#111111"

WS = LS = 41.0
WP, LP = 40.0, 29.0
SW, SL = 6.8, 1.0           # slot width (horizontal) and length (vertical)
WF, LF = 3.0, 6.0           # feed: schematic

PX0 = (WS - WP) / 2
PY0 = 6.0                   # patch bottom edge; feed runs from substrate edge to here
PY1 = PY0 + LP
SY = PY1 - 2.6              # slot near the radiating (top) edge


def dim(ax, p0, p1, label, off, fs=12):
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=1.1, shrinkA=0, shrinkB=0,
                                mutation_scale=10))
    ax.text((p0[0] + p1[0]) / 2 + off[0], (p0[1] + p1[1]) / 2 + off[1], label,
            ha="center", va="center", fontsize=fs, fontweight="bold", color=INK)


def main():
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                         "mathtext.bf": "Arial:bold",
                         "savefig.bbox": "tight", "savefig.pad_inches": 0.03})
    fig, ax = plt.subplots(figsize=(3.3, 3.2))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-8, WS + 8)
    ax.set_ylim(-8, LS + 2)

    ax.add_patch(Rectangle((0, 0), WS, LS, fc=SUB, ec=EDGE_S, lw=1.0))
    ax.add_patch(Rectangle((PX0, PY0), WP, LP, fc=PATCH, ec=EDGE_P, lw=1.0))
    ax.add_patch(Rectangle((WS / 2 - WF / 2, 0), WF, PY0, fc=PATCH, ec=EDGE_P, lw=1.0))
    ax.add_patch(Rectangle((WS / 2 - SW / 2, SY), SW, SL, fc=SLOT, ec=EDGE_S, lw=0.8))

    # substrate
    dim(ax, (-3.5, 0), (-3.5, LS), r"$\mathbf{L_s}$", (-2.6, 0), fs=14)
    dim(ax, (0, -3.2), (WS, -3.2), r"$\mathbf{W_s}$", (0, -2.6), fs=14)
    # patch
    dim(ax, (PX0 + 0.2, PY0 + LP * 0.55), (PX0 + WP - 0.2, PY0 + LP * 0.55), r"$\mathbf{W_p}$",
        (0, 2.2), fs=14)
    dim(ax, (WS + 2.5, PY0), (WS + 2.5, PY1), r"$\mathbf{L_p}$", (2.9, 0), fs=14)
    # slot
    dim(ax, (WS / 2 - SW / 2, SY - 1.6), (WS / 2 + SW / 2, SY - 1.6), "", (0, 0))
    ax.text(WS / 2 - SW / 2 - 2.6, SY + SL / 2, r"$\mathbf{S_w}$", ha="center", va="center",
            fontsize=11, fontweight="bold")
    ax.text(WS / 2 + SW / 2 + 2.6, SY + SL / 2, r"$\mathbf{S_L}$", ha="center", va="center",
            fontsize=11, fontweight="bold")
    # feed
    ax.text(WS / 2 - 0.2, PY0 + 2.6, r"$\mathbf{W_f}$", ha="center", va="center", fontsize=11)
    dim(ax, (WS / 2 + WF / 2 + 1.0, 0.2), (WS / 2 + WF / 2 + 1.0, PY0 - 0.2), r"$\mathbf{L_f}$",
        (2.0, 0), fs=11)

    fig.savefig(OUT, dpi=400)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
