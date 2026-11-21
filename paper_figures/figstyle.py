"""Shared plot style for the paper figures (solver-export look: boxed axes, inward ticks, grid).

One text size and one figure size for every plot, so that at a common placed width all plots
print with the same text size.
"""
import os

import matplotlib.pyplot as plt

FONT_PT = float(os.environ.get("FIG_FONT_PT", 5.7))
FIGSIZE = (3.4, 2.6)          # aspect 1.31
LW = 0.9                      # data-line width


def origin_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": FONT_PT,
        "axes.labelsize": FONT_PT,
        "axes.titlesize": 10,
        "axes.linewidth": 0.7,
        "axes.edgecolor": "black",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.grid": True,
        "grid.color": "#B0B0B0",
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 2.5,
        "xtick.labelsize": FONT_PT,
        "ytick.major.size": 2.5,
        "ytick.labelsize": FONT_PT,
        "xtick.minor.visible": False,
        "legend.frameon": True,
        "legend.edgecolor": "black",
        "legend.fancybox": False,
        "legend.framealpha": 1.0,
        "legend.fontsize": FONT_PT,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "mathtext.default": "regular",
    })
