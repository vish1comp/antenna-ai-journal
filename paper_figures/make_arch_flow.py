"""Two compact single-column figures that replace the full-width pipeline figure:

  figures/architecture.png  - layered block diagram of the system (what the parts are)
  figures/flowchart.png     - step-by-step flow chart with the two decisions (how it runs)

The detailed full-width diagram (make_architecture_fig.py) is kept as figures/pipeline_detailed.png.

    python make_arch_flow.py
"""
from __future__ import annotations

import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
INK = "#1A1A1A"
# One colour throughout (panel preference); line style, not hue, separates the paths.
BLUE = "#2B5C9E"
TEAL = ORANGE = GREEN = PURPLE = BLUE
LANE = "#EEF3F8"
TITLE, BODY = 6.6, 5.9


def setup(w, h, xmax, ymax):
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.axis("off")
    return fig, ax


def box(ax, cx, cy, w, h, title, body, colour, rounding=0.12):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={rounding}",
                                facecolor="white", edgecolor=colour, linewidth=1.1, zorder=2))
    if body:
        ax.text(cx, cy + h * 0.2, title, ha="center", va="center", fontsize=TITLE, fontweight="bold",
                color=colour, zorder=3)
        ax.text(cx, cy - h * 0.2, body, ha="center", va="center", fontsize=BODY, color=INK, zorder=3,
                linespacing=1.2)
    else:
        ax.text(cx, cy, title, ha="center", va="center", fontsize=TITLE, fontweight="bold", color=colour,
                zorder=3, linespacing=1.2)


def arrow(ax, p0, p1, colour=INK, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, color=colour, linewidth=0.9,
                                 linestyle=ls, shrinkA=0, shrinkB=0, zorder=4))


def path(ax, pts, colour=INK, ls="-"):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color=colour, lw=0.9, ls=ls, zorder=4, solid_capstyle="butt")
    arrow(ax, pts[-2], pts[-1], colour, ls)


def note(ax, x, y, text, colour, ha="center", rot=0):
    ax.text(x, y, text, ha=ha, va="center", fontsize=BODY - 0.4, style="italic", color=colour, rotation=rot,
            zorder=5, bbox=dict(boxstyle="round,pad=0.08", facecolor="white", edgecolor="none"))


# ------------------------------------------------------------------ architecture
def architecture():
    fig, ax = setup(3.5, 2.75, 10.4, 8.2)
    lanes = [("DATA", 6.25, LANE), ("MODEL", 4.3, LANE), ("DESIGN", 2.35, LANE), ("OUTPUT", 0.4, LANE)]
    for name, y, col in lanes:
        ax.add_patch(Rectangle((0.0, y), 10.0, 1.6, facecolor=col, edgecolor="none", zorder=0))
        ax.add_patch(Rectangle((0.0, y), 0.75, 1.6, facecolor="#D5DCE3", edgecolor="none", zorder=0))
        ax.text(0.375, y + 0.8, name, rotation=90, ha="center", va="center", fontsize=BODY, fontweight="bold",
                color="#33475B")
    L, R, W = 3.3, 7.65, 4.0                   # left / right column centres, box width
    box(ax, L, 7.05, W, 1.25, "Patch-antenna datasets", "A: CST · B: rect. · C: TL model", BLUE)
    box(ax, R, 7.05, W, 1.25, "Physics-informed curation", "resonance in the target band", GREEN)
    box(ax, L, 5.1, W, 1.25, "Forward surrogates", "MLP  ·  GBDT  ·  RF  ·  GP", BLUE)
    box(ax, R, 5.1, W, 1.25, "Trust layer", "GBDT ensemble µ, σ + conformal", TEAL)
    box(ax, L, 3.15, W, 1.25, "Optimizers", "PSO · GA · BO  (spec / risk-averse)", ORANGE)
    box(ax, R, 3.15, W, 1.25, "Conditional generators", "GMM · VAE · diffusion + CFG", PURPLE)
    box(ax, 5.475, 1.2, 8.35, 1.25, "Trusted designs", "verified designs  ·  90% intervals  ·  design portfolio",
        GREEN)
    arrow(ax, (L, 6.42), (L, 5.73))                                   # data -> surrogates
    arrow(ax, (L + W / 2, 5.1), (R - W / 2, 5.1))                     # surrogates -> trust
    arrow(ax, (L, 4.47), (L, 3.78))                                   # surrogates -> optimizers
    arrow(ax, (R - 0.9, 4.47), (L + 1.1, 3.78), TEAL)                 # trust -> optimizers (µ, σ)
    arrow(ax, (R, 3.78), (R, 4.47), PURPLE, ls=(0, (3, 1.5)))         # generators scored by trust layer
    note(ax, R + 0.12, 4.13, "scored", PURPLE, ha="left")
    note(ax, 5.47, 4.0, "µ, σ", TEAL)
    path(ax, [(R + W / 2, 7.05), (10.2, 7.05), (10.2, 3.15), (R + W / 2, 3.15)], GREEN, ls=(0, (3, 1.5)))
    arrow(ax, (L, 2.52), (L, 1.83))                                   # optimizers -> output
    arrow(ax, (R, 2.52), (R, 1.83))                                   # generators -> output
    out = os.path.join(FIG, "architecture.png")
    fig.savefig(out, dpi=600)
    plt.close(fig)
    print("wrote", out)


# ------------------------------------------------------------------ flow chart
def diamond(ax, cx, cy, w, h, text, colour):
    ax.add_patch(Polygon([[cx - w / 2, cy], [cx, cy + h / 2], [cx + w / 2, cy], [cx, cy - h / 2]], closed=True,
                         facecolor="white", edgecolor=colour, linewidth=1.1, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=TITLE, fontweight="bold", color=colour, zorder=3,
            linespacing=1.1)


def terminal(ax, cx, cy, w, h, text, colour):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={h / 2}",
                                facecolor=colour, edgecolor=colour, linewidth=1.0, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=TITLE, fontweight="bold", color="white", zorder=3)


def flowchart():
    fig, ax = setup(3.5, 4.1, 10.0, 11.7)
    C, W, H = 5.0, 5.6, 0.72
    terminal(ax, C, 11.2, 3.6, 0.6, "Start: antenna dataset", BLUE)
    box(ax, C, 10.15, W, H, "Pre-process and split (fixed seed)", None, BLUE)
    box(ax, C, 9.05, W, H, "Train surrogates and GBDT ensemble", None, BLUE)
    box(ax, C, 7.95, W, H, "Split-conformal calibration (90%)", None, TEAL)
    diamond(ax, C, 6.7, 3.2, 1.25, "Design\nmode?", BLUE)
    LX, RX, BW = 2.55, 7.45, 3.6
    box(ax, LX, 5.15, BW, 1.0, "Optimize", "PSO · GA · BO\nspec mask or µ + κσ", ORANGE)
    box(ax, RX, 5.15, BW, 1.0, "Generate", "GMM · VAE · diffusion\nguidance scale w", PURPLE)
    box(ax, C, 3.75, W, H, "Score candidates: µ, σ, interval", None, TEAL)
    diamond(ax, C, 2.5, 3.2, 1.25, "Trusted?", TEAL)
    box(ax, C, 1.2, W, H, "Verify: nearest real design / exact physics", None, GREEN)
    terminal(ax, C, 0.3, 3.6, 0.6, "Output trusted designs", GREEN)

    for y0, y1 in ((10.9, 10.51), (9.79, 9.41), (8.69, 8.31), (7.59, 7.33), (2.5 - 0.625, 1.56)):
        arrow(ax, (C, y0), (C, y1))
    arrow(ax, (C, 0.84), (C, 0.6))
    # design mode -> optimize / generate
    path(ax, [(C - 1.6, 6.7), (LX, 6.7), (LX, 5.65)])
    path(ax, [(C + 1.6, 6.7), (RX, 6.7), (RX, 5.65)])
    note(ax, LX + 0.65, 6.92, "optimize", ORANGE)
    note(ax, RX - 0.65, 6.92, "generate", PURPLE)
    # both -> score
    arrow(ax, (LX, 4.65), (LX, 4.11))
    arrow(ax, (RX, 4.65), (RX, 4.11))
    arrow(ax, (C, 3.39), (C, 3.125))
    note(ax, C + 0.3, 1.86, "yes", GREEN, ha="left")
    # no -> flag and re-design with the risk-averse objective
    path(ax, [(C - 1.6, 2.5), (0.3, 2.5), (0.3, 5.15), (LX - BW / 2, 5.15)], ORANGE, ls=(0, (3, 1.5)))
    note(ax, 0.3, 3.85, "no: flag, re-design", ORANGE, rot=90)
    # curated training data (chosen at pre-processing) feeds the generator
    path(ax, [(C + W / 2, 10.15), (9.75, 10.15), (9.75, 5.15), (RX + BW / 2, 5.15)], GREEN, ls=(0, (3, 1.5)))
    note(ax, 9.75, 7.6, "curated training data", GREEN, rot=90)
    out = os.path.join(FIG, "flowchart.png")
    fig.savefig(out, dpi=600)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    full = os.path.join(FIG, "pipeline.png")
    if os.path.exists(full):
        shutil.copy(full, os.path.join(FIG, "pipeline_detailed.png"))
    architecture()
    flowchart()
