"""Figures for the improved paper's new experiments, in the same Origin style as
make_origin_figs.py.  Data: ../results/improved/.

    python make_ext_figs.py            # uq_baselines.png, inv_baselines.png, b_learning_curve.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
import figstyle as F            # noqa: E402

R = os.path.join(HERE, "..", "results", "improved")
FIG = os.path.join(HERE, "figures")
LW = F.LW
# one fixed colour per method, used in every panel
UQ = [("GBDT ensemble (ours)", "Bootstrap GBDT ens. (ours)", "#0000FF", "o"),
      ("MLP deep ensemble", "MLP deep ensemble", "#FF0000", "s"),
      ("MC dropout", "MC dropout", "#FF8C00", "^"),
      ("CQR (quantile GBDT)", "CQR", "#800080", "D"),
      ("GP", "Gaussian process", "#7F7F7F", "v")]


def save(fig, name):
    out = os.path.join(FIG, name)
    fig.savefig(out, dpi=600)
    plt.close(fig)
    print("wrote", out)


def fig_uq():
    d = pd.read_csv(os.path.join(R, "uq_baselines.csv"))
    g = d.groupby(["dataset", "method"])
    fig, (a, b) = plt.subplots(2, 1, figsize=(F.FIGSIZE[0], F.FIGSIZE[1] * 2 + 0.6),
                               gridspec_kw={"hspace": 0.38, "top": 0.86})
    x = np.arange(3)
    bw = 0.16
    for k, (m, lab, col, mk) in enumerate(UQ):
        off = (k - 2) * bw
        raw = [g.get_group((ds, m))["cov_raw"] for ds in "ABC"]
        cal = [g.get_group((ds, m))["cov_conf"] for ds in "ABC"]
        wid = [g.get_group((ds, m))["width_conf"] for ds in "ABC"]
        a.errorbar(x + off, [100 * r.mean() for r in raw], yerr=[100 * r.std() for r in raw], fmt=mk,
                   color=col, mfc="white", mec=col, ms=3.2, lw=LW, capsize=1.5, zorder=3)
        a.errorbar(x + off, [100 * c.mean() for c in cal], yerr=[100 * c.std() for c in cal], fmt=mk,
                   color=col, mfc=col, mec="black", mew=0.4, ms=3.2, lw=LW, capsize=1.5, zorder=3, label=lab)
        b.bar(x + off, [w.mean() for w in wid], bw, yerr=[w.std() for w in wid], color=col,
              edgecolor="black", lw=0.5, error_kw=dict(lw=0.6, capsize=1.2), zorder=2, label=lab)
    a.axhline(90, color="black", ls="--", lw=LW, zorder=1)
    a.set_ylim(55, 102)
    a.set_ylabel("Coverage of 90% interval (%)")
    a.set_xticks(x, ["Dataset A", "Dataset B", "Dataset C"])
    a.set_title("(a)", loc="left")
    a.set_title("open: raw    filled: conformal", loc="right")
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0), columnspacing=0.8, handletextpad=0.2)
    b.set_ylabel("Calibrated interval width (dB)")
    b.set_xticks(x, ["Dataset A", "Dataset B", "Dataset C"])
    b.set_title("(b)", loc="left")
    b.set_ylim(0, 3.6)
    save(fig, "uq_baselines.png")


def fig_bsize():
    p = os.path.join(R, "b_learning_curve.csv")
    d = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=F.FIGSIZE)
    for m, col, mk in (("MLP", "#FF0000", "s"), ("GBDT", "#0000FF", "o")):
        s = d[d.model == m].groupby("n_train")["R2"].agg(["mean", "std"]).fillna(0)
        ax.errorbar(s.index, s["mean"], yerr=s["std"], fmt="-" + mk, color=col, mfc=col, mec="black",
                    mew=0.4, ms=3.2, lw=LW, capsize=1.5, label=m)
    ax.axvline(6000, color="black", ls=":", lw=LW)
    ax.text(6600, 0.06, "6000 rows")
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Training rows (dataset B)")
    ax.set_ylabel("Test $R^2$")
    ax.legend(loc="lower right")
    save(fig, "b_learning_curve.png")


if __name__ == "__main__":
    F.origin_style()
    for f in sys.argv[1:] or ["uq", "bsize"]:
        globals()["fig_" + f]()
