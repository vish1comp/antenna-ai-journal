"""Figs. 3, 4 and 5 of the journal paper in the MAPCON "Origin" plot style, so they
match the HFSS/CST reflection-coefficient figure: Arial at the solver-export text
size, black frame with inward ticks on all four sides, light grey full grid, plain
boxed legend, thin lines in black (dashed) / red / orange / blue / purple, and the
same 3.4 x 2.6 in canvas. The style comes straight from
figstyle.py (origin_style, FONT_PT, FIGSIZE, LW).

Data are the regenerated arrays (paper seed) in ../results/:
spec_curves.npz, uq_data.npz, gen_compare.npz.

    python make_origin_figs.py
"""
from __future__ import annotations

import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import figstyle as F            # noqa: E402

RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "figures")
# the S11 solver plot's own colours
BLACK, RED, ORANGE, BLUE, PURPLE, GREEN = "black", "#FF0000", "#FF8C00", "#0000FF", "#800080", "#108A2E"
LW = F.LW


def save(fig, name):
    out = os.path.join(FIG, name)
    keep = out.replace(".png", "_v2.png")          # the previous redraw, kept for reference
    if os.path.exists(out) and not os.path.exists(keep) and os.path.exists(out.replace(".png", "_v1.png")):
        shutil.copy(out, keep)
    fig.savefig(out, dpi=600)
    plt.close(fig)
    print("wrote", out)


def conformal_q(res, alpha):
    n = len(res)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(res, level, method="higher"))


# ---------------------------------------------------------------- Fig. 3
def fig_spec():
    d = np.load(os.path.join(RES, "spec_curves.npz"))
    fig, ax = plt.subplots(figsize=F.FIGSIZE)
    ax.axvspan(*d["band"], color="#E6E6E6", lw=0, zorder=0, label="Target band")
    ax.axhline(float(d["mask"]), color=RED, ls="--", lw=LW, label="−10 dB mask")
    ax.plot(d["true_freqs"], d["true_s11"], color=BLACK, ls="--", lw=LW, label="Nearest real design")
    ax.plot(d["freqs"], d["s11"], color=BLUE, ls="-", lw=LW, label="Designed (surrogate)")
    ax.set_xlim(2.2, 2.6)
    ax.set_ylim(-30, 0)
    ax.set_xticks(np.arange(2.2, 2.61, 0.1))
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Reflection Coefficient (dB)")
    ax.legend(loc="lower right")
    save(fig, "spec_design_response.png")


# ---------------------------------------------------------------- Fig. 4
def fig_uq():
    d = np.load(os.path.join(RES, "uq_data.npz"))
    yc, mc, sc = d["y_cal"], d["mu_cal"], np.maximum(d["sd_cal"], 1e-9)
    yt, mt, st = d["y_test"], d["mu_test"], np.maximum(d["sd_test"], 1e-9)
    err = np.abs(yt - mt)
    fig, (a, b) = plt.subplots(2, 1, figsize=(F.FIGSIZE[0], F.FIGSIZE[1] * 2 + 0.35),
                               gridspec_kw={"hspace": 0.45})

    # (a) reliability: every test point, binned RMSE, ideal line
    a.scatter(st, np.clip(err, 1e-3, None), s=2, color=BLUE, alpha=0.18, edgecolors="none",
              label="Test points", rasterized=True)
    order = np.argsort(st)
    bins = np.array_split(order, 15)
    xs = [st[k].mean() for k in bins]
    ys = [np.sqrt((err[k] ** 2).mean()) for k in bins]
    a.plot(xs, ys, "-o", color=RED, lw=LW, ms=2.5, label="Binned RMSE")
    g = np.array([1e-3, 6])
    a.plot(g, g, "--", color=BLACK, lw=LW, label="Ideal (RMSE = σ)")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlim(1e-3, 6)
    a.set_ylim(1e-3, 40)
    a.set_xlabel("Ensemble Spread σ (dB)")
    a.set_ylabel("|Prediction Error| (dB)")
    a.legend(loc="upper left")
    a.text(0.5, -0.32, "(a)", transform=a.transAxes, ha="center", va="top")

    # (b) calibration: coverage vs nominal level
    levels = np.linspace(0.5, 0.99, 50)
    res_c = np.abs(yc - mc)
    cov_f = [np.mean(err <= conformal_q(res_c, 1 - L)) for L in levels]
    cov_n = [np.mean(err <= conformal_q(res_c / sc, 1 - L) * st) for L in levels]
    b.plot([0.5, 1.0], [0.5, 1.0], "--", color=BLACK, lw=LW, label="Ideal")
    b.plot(levels, cov_f, "-", color=BLUE, lw=LW, label="Constant width")
    b.plot(levels, cov_n, "-", color=RED, lw=LW, label="σ-scaled")
    b.plot([0.9], [float(d["cov"])], "o", color=BLUE, ms=2.8)
    b.plot([0.9], [float(d["covn"])], "s", color=RED, ms=2.5, mfc="none", mew=0.7)
    b.set_xlim(0.5, 1.0)
    b.set_ylim(0.5, 1.0)
    ticks = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    b.set_xticks(ticks)
    b.set_yticks(ticks)
    b.set_xticklabels([f"{round(v * 100)}" for v in ticks])
    b.set_yticklabels([f"{round(v * 100)}" for v in ticks])
    b.set_xlabel("Nominal Coverage (%)")
    b.set_ylabel("Test Coverage (%)")
    b.legend(loc="lower right")
    b.text(0.5, -0.32, "(b)", transform=b.transAxes, ha="center", va="top")
    fig.align_ylabels([a, b])
    save(fig, "uq_reliability.png")


# ---------------------------------------------------------------- Fig. 5
def fig_gen():
    from scipy.stats import gaussian_kde
    d = np.load(os.path.join(RES, "gen_compare.npz"))
    fig, ax = plt.subplots(figsize=F.FIGSIZE)
    x = np.linspace(-37, 0, 800)
    for key, name, col in (("gmm", "Conditional GMM", BLUE), ("vae", "Conditional VAE", RED),
                           ("diffusion", "Conditional Diffusion", GREEN)):
        s = np.asarray(d[key], float)
        # smoothed distribution of the 400 designs (kernel density, % of designs per dB);
        # bandwidth kept small so a curve does not spill past the mask where no design lies
        y = gaussian_kde(s, bw_method=0.09)(x) * 100
        ax.plot(x, y, color=col, lw=LW + 0.2, label=name)
        ax.fill_between(x, y, color=col, alpha=0.08, lw=0)
    ax.axvline(float(d["target"]), color=PURPLE, ls="-", lw=LW, label="Target (−20 dB)")
    ax.axvline(float(d["mask"]), color=BLACK, ls="--", lw=LW, label="Mask (−10 dB)")
    ax.set_xlim(-37, 0)
    ax.set_ylim(0, None)
    ax.set_xlabel("Achieved S11 at 2.4 GHz (dB)")
    ax.set_ylabel("Share of Designs (% per dB)")
    ax.legend(loc="upper right")
    save(fig, "generative_compare_fidelity.png")


# ---------------------------------------------------------------- Fig. 6
def fig_guidance():
    d = np.load(os.path.join(RES, "guidance.npz"))
    w, fid, div = d["w"], d["below_mask"] * 100, d["diversity"]
    fig, ax = plt.subplots(figsize=F.FIGSIZE)
    ax.tick_params(right=False)                      # the right side belongs to the twin axis
    l1, = ax.plot(w, fid, "-o", color=BLUE, lw=LW, ms=3, label="Designs below −10 dB mask")
    ax.set_xlabel("Guidance Scale w")
    ax.set_ylabel("Designs below −10 dB Mask (%)", color=BLUE)
    ax.tick_params(axis="y", colors=BLUE)
    ax.set_xlim(-0.2, 5.2)
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_ylim(75, 101)
    ax2 = ax.twinx()
    ax2.grid(False)
    l2, = ax2.plot(w, div, "--s", color=RED, lw=LW, ms=2.8, label="Design diversity")
    ax2.set_ylabel("Design Diversity", color=RED)
    ax2.tick_params(axis="y", colors=RED, direction="in")
    ax2.set_ylim(0.05, 0.28)
    # the lower middle is empty once both axes leave headroom below the curves
    ax.legend(handles=[l1, l2], loc="lower center")
    save(fig, "generative_guidance.png")


# ---------------------------------------------------------------- Fig. 7
def fig_cross():
    import pandas as pd
    raw = pd.read_csv(os.path.join(RES, "cross_dataset_raw.csv"))
    sets = sorted(raw["Dataset"].unique())                 # A., B., C.
    names = ["A: CST patch\n(real)", "B: Rect. patch\n(real)", "C: TL model\n(synthetic)"]
    # Origin-style symbols: one shape and colour per surrogate
    models = [("MLP", BLUE, "o"), ("GBDT", RED, "s"), ("Random Forest", GREEN, "^"), ("GP", ORANGE, "D")]
    # wider than tall, like the bar chart it replaces, so the page layout is unchanged
    fig, ax = plt.subplots(figsize=(F.FIGSIZE[0], 2.0))
    ax.grid(axis="x", visible=False)
    xs = np.arange(len(sets))
    for j in xs:                                           # alternate shaded dataset panels
        if j % 2 == 0:
            ax.axvspan(j - 0.5, j + 0.5, color="#F0F0F0", lw=0, zorder=0)
    step = 0.2
    for i, (m, col, mk) in enumerate(models):
        off = (i - 1.5) * step
        mus = []
        for j, s in enumerate(sets):
            r2 = raw[(raw["Dataset"] == s) & (raw["Model"] == m)]["R2"].to_numpy()
            mu, sd = r2.mean(), r2.std(ddof=1)
            mus.append(mu)
            # light dots for the three single splits, slightly jittered so they do not overlap
            ax.scatter(xs[j] + off + np.array([-0.035, 0, 0.035]), r2, s=3, color=col, alpha=0.35,
                       edgecolors="none", zorder=2)
            ax.errorbar(xs[j] + off, mu, yerr=sd, fmt=mk, color=col, ms=3.4, mfc=col, mec="black",
                        mew=0.4, lw=0.8, capsize=1.8, capthick=0.8, zorder=3,
                        label=m if j == 0 else None)
    ax.set_xticks(xs)
    ax.set_xticklabels(names)
    ax.tick_params(axis="x", which="both", top=False, bottom=False)
    ax.set_xlim(-0.5, len(sets) - 0.5)
    ax.set_ylim(0, 1.2)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("Test R² (mean ± std, 3 splits)")
    ax.legend(loc="upper center", ncol=4, columnspacing=0.8, handlelength=1.4, handletextpad=0.3)
    save(fig, "cross_dataset.png")


def _cross_stats():
    import pandas as pd
    raw = pd.read_csv(os.path.join(RES, "cross_dataset_raw.csv"))
    sets = sorted(raw["Dataset"].unique())
    models = ["MLP", "GBDT", "Random Forest", "GP"]
    mu = np.array([[raw[(raw.Dataset == s) & (raw.Model == m)].R2.mean() for s in sets] for m in models])
    sd = np.array([[raw[(raw.Dataset == s) & (raw.Model == m)].R2.std(ddof=1) for s in sets] for m in models])
    return models, mu, sd


def fig_cross_heatmap(name="cross_dataset_heatmap.png"):
    """Annotated grid: rows = surrogates, columns = datasets, cell = mean R2 (± std)."""
    from matplotlib.colors import LinearSegmentedColormap
    models, mu, sd = _cross_stats()
    cols = ["A: CST patch\n(real)", "B: Rect. patch\n(real)", "C: TL model\n(synthetic)"]
    cmap = LinearSegmentedColormap.from_list("rb", ["#FF0000", "#FF8C00", "#FFF2A8", "#7FB3FF", "#0000FF"])
    fig, ax = plt.subplots(figsize=(F.FIGSIZE[0], 2.0))
    ax.grid(False)
    im = ax.imshow(mu, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(mu.shape[0]):
        for j in range(mu.shape[1]):
            v = mu[i, j]
            ink = "white" if (v < 0.22 or v > 0.82) else "black"
            best = i == np.argmax(mu[:, j])
            ax.text(j, i - 0.1, f"{v:.3f}", ha="center", va="center", color=ink,
                    fontsize=F.FONT_PT + 0.6, fontweight="bold" if best else "normal")
            ax.text(j, i + 0.24, f"± {sd[i, j]:.3f}", ha="center", va="center", color=ink,
                    fontsize=F.FONT_PT - 1.4)
    ax.set_xticks(range(3)); ax.set_xticklabels(cols)
    ax.set_yticks(range(4)); ax.set_yticklabels(models)
    ax.tick_params(which="both", length=0, top=False, right=False, bottom=False, left=False)
    ax.set_xticks(np.arange(-0.5, 3), minor=True); ax.set_yticks(np.arange(-0.5, 4), minor=True)
    ax.grid(which="minor", color="white", lw=1.2)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("Test R² (mean of 3 splits)")
    cb.ax.tick_params(direction="in", length=2)
    cb.outline.set_linewidth(0.6)
    save(fig, name)


def fig_cross_hbar(name="cross_dataset_hbar.png"):
    """Horizontal grouped bars, one panel per dataset, with hatch patterns."""
    models, mu, sd = _cross_stats()
    titles = ["A: CST patch (real)", "B: Rect. patch (real)", "C: TL model (synthetic)"]
    colors = [BLUE, RED, GREEN, ORANGE]
    hatches = ["", "////", "\\\\\\\\", "xxx"]
    fig, axes = plt.subplots(1, 3, figsize=(F.FIGSIZE[0], 2.0), sharey=True, gridspec_kw={"wspace": 0.12})
    y = np.arange(len(models))[::-1]
    for j, ax in enumerate(axes):
        ax.grid(axis="y", visible=False)
        for i in range(len(models)):
            ax.barh(y[i], mu[i, j], 0.68, xerr=sd[i, j], color="white", edgecolor=colors[i],
                    hatch=hatches[i], lw=0.8, error_kw=dict(lw=0.6, capsize=1.4, ecolor="black"))
            # value sits inside the bar near its base, on a white patch over the hatching
            ax.text(0.04, y[i], f"{mu[i, j]:.2f}", va="center", fontsize=F.FONT_PT - 1.2,
                    color=colors[i], fontweight="bold",
                    bbox=dict(fc="white", ec="none", pad=0.6), zorder=5)
        ax.set_xlim(0, 1.05); ax.set_xticks([0, 0.5, 1.0]); ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_title(titles[j], fontsize=F.FONT_PT - 0.6, pad=3)
    axes[0].set_yticks(y); axes[0].set_yticklabels(models)
    axes[1].set_xlabel("Test R² (mean ± std, 3 splits)")
    save(fig, name)


# ---------------------------------------------------------------- Fig. 8
def fig_curated():
    """Broad -> curated training, as arrows: (a) share under the mask, (b) median S11."""
    import pandas as pd
    df = pd.read_csv(os.path.join(RES, "generative_curated.csv"))
    methods = [("GMM", "GMM"), ("Diffusion+CFG", "Diffusion\n+ CFG")]
    get = lambda data, m, col: float(df[(df["data"] == data) & (df["method"] == m)][col].iloc[0])
    fig, (a, b) = plt.subplots(1, 2, figsize=(F.FIGSIZE[0], 1.75), sharey=True,
                               gridspec_kw={"wspace": 0.16})
    y = np.array([1.0, 0.0])
    panels = ((a, "below_mask", "Designs below −10 dB Mask (%)", (0, 100), "{:+.1f} pts", "(a)"),
              (b, "median", "Median S11 (dB)", (-14, -4), "{:+.1f} dB", "(b)"))
    for ax, col, xlab, xl, fmt, tag in panels:
        ax.grid(axis="y", visible=False)
        for yi, (m, _) in zip(y, methods):
            v0, v1 = get("broad", m, col), get("curated", m, col)
            ax.annotate("", xy=(v1, yi), xytext=(v0, yi),
                        arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.0, mutation_scale=6,
                                        shrinkA=2.5, shrinkB=3.2))
            ax.plot(v0, yi, "o", ms=4, mfc="white", mec=BLACK, mew=0.8, zorder=3)
            ax.plot(v1, yi, "o", ms=4, mfc=GREEN, mec=BLACK, mew=0.5, zorder=3)
            ax.text((v0 + v1) / 2, yi + 0.2, fmt.format(v1 - v0).replace("-", "−"), ha="center",
                    va="bottom", fontsize=F.FONT_PT - 0.8, color=GREEN)
            for v in (v0, v1):
                ax.text(v, yi - 0.2, f"{v:.1f}".replace("-", "−"), ha="center", va="top",
                        fontsize=F.FONT_PT - 1.4, zorder=4,
                        bbox=dict(fc="white", ec="none", pad=0.3))   # readable over the mask line
        ax.set_xlim(*xl)
        ax.set_xlabel(xlab)
        ax.text(0.5, -0.4, tag, transform=ax.transAxes, ha="center", va="top")
    a.set_xticks([0, 20, 40, 60, 80, 100])
    b.set_xlim(-3.5, -14)                               # no tick at the left edge, so "100" stands alone
    b.axvline(-10, color=RED, ls="--", lw=LW, zorder=1)
    b.text(-10.2, 1.6, "−10 dB mask", color=RED, ha="left", va="center", fontsize=F.FONT_PT - 1.4)
    b.set_xticks([-6, -8, -10, -12, -14])
    a.set_ylim(-0.6, 1.75)
    a.set_yticks(y)
    a.set_yticklabels([lab for _, lab in methods])
    a.tick_params(axis="y", length=0)
    b.tick_params(axis="y", length=0)
    h = [plt.Line2D([], [], ls="", marker="o", ms=4, mfc="white", mec=BLACK, mew=0.8, label="Broad training"),
         plt.Line2D([], [], ls="", marker="o", ms=4, mfc=GREEN, mec=BLACK, mew=0.5, label="Physics-curated")]
    fig.legend(handles=h, loc="upper center", ncol=2, bbox_to_anchor=(0.55, 1.0))
    fig.subplots_adjust(top=0.86, bottom=0.33)
    save(fig, "generative_curated.png")


def fig_curated_simple():
    """One panel: share of designs meeting the mask, broad vs curated training, per model."""
    import pandas as pd
    df = pd.read_csv(os.path.join(RES, "generative_curated.csv"))
    methods = [("GMM", "GMM"), ("Diffusion+CFG", "Guided Diffusion")]
    get = lambda data, m: float(df[(df["data"] == data) & (df["method"] == m)]["below_mask"].iloc[0])
    broad = [get("broad", m) for m, _ in methods]
    cur = [get("curated", m) for m, _ in methods]
    fig, ax = plt.subplots(figsize=(F.FIGSIZE[0], 2.2))
    ax.grid(axis="x", visible=False)
    x = np.arange(len(methods))
    bw = 0.32
    b1 = ax.bar(x - bw / 2, broad, bw, color="#BFBFBF", edgecolor="black", lw=0.6, label="Broad training", zorder=2)
    b2 = ax.bar(x + bw / 2, cur, bw, color=GREEN, edgecolor="black", lw=0.6, label="Physics-curated training", zorder=2)
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1.5, f"{r.get_height():.1f}%",
                    ha="center", va="bottom", fontsize=F.FONT_PT, fontweight="bold")
    # the gain, as a bracket over each pair
    for xi, v0, v1 in zip(x, broad, cur):
        top = v1 + 13
        ax.plot([xi - bw / 2, xi - bw / 2, xi + bw / 2, xi + bw / 2], [v0 + 9.5, top, top, v1 + 9.5],
                color=RED, lw=0.7)
        ax.text(xi, top + 1.2, f"+{v1 - v0:.1f} pts", ha="center", va="bottom", color=RED,
                fontsize=F.FONT_PT)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in methods])
    ax.tick_params(axis="x", which="both", top=False, bottom=False)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Designs Meeting −10 dB Mask (%)")
    ax.legend(loc="upper left")
    save(fig, "generative_curated.png")


# ---------------------------------------------------------------- multi-seed figures
JR = RES


def _ms(name):
    import pandas as pd
    return pd.read_csv(os.path.join(JR, name))


def fig_cov_seeds():
    """Coverage of the 90% intervals for every seed: conformal (fixed, scaled) vs raw Gaussian."""
    df = _ms("ms_uq.csv")
    fig, ax = plt.subplots(figsize=(F.FIGSIZE[0], 2.1))
    ax.grid(axis="x", visible=False)
    cols = [("cov_raw", "Raw ensemble\nμ ± 1.645σ", BLACK, "D"), ("cov_fixed", "Conformal\nconstant width", BLUE, "o"),
            ("cov_scaled", "Conformal\nσ-scaled", RED, "s")]
    rng = np.random.default_rng(0)
    for i, (c, lab, col, mk) in enumerate(cols):
        v = df[c].to_numpy() * 100
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=9, color=col, marker=mk, alpha=0.75,
                   edgecolors="none", zorder=3)
        ax.errorbar(i + 0.25, v.mean(), yerr=v.std(ddof=1), fmt=mk, color=col, ms=3.5, mfc="white",
                    mew=0.8, lw=0.8, capsize=2, zorder=4)
    ax.axhline(90, color=GREEN, ls="--", lw=LW, zorder=2)
    ax.text(0.5, 88.6, "target 90%", color=GREEN, ha="center", va="top", fontsize=F.FONT_PT - 1)
    ax.set_ylim(55, 95)
    ax.set_xticks(range(3))
    ax.set_xticklabels([c[1] for c in cols])
    ax.tick_params(axis="x", which="both", top=False, bottom=False)
    ax.set_xlim(-0.5, 2.6)
    ax.set_ylabel("Test Coverage (%)")
    save(fig, "uq_coverage_seeds.png")


def fig_ensemble_size():
    df = _ms("ms_ensemble.csv")
    g = df.groupby("M").agg(["mean", "std"])
    M = g.index.to_numpy()
    fig, ax = plt.subplots(figsize=(F.FIGSIZE[0], 2.1))
    ax.tick_params(right=False)
    l1 = ax.errorbar(M, g[("cov_scaled", "mean")] * 100, yerr=g[("cov_scaled", "std")] * 100, fmt="-o",
                     color=BLUE, lw=LW, ms=3, capsize=2, label="Coverage (σ-scaled)")
    ax.axhline(90, color=BLUE, ls=":", lw=0.7)
    ax.set_xlabel("Ensemble Size M")
    ax.set_ylabel("Test Coverage (%)", color=BLUE)
    ax.tick_params(axis="y", colors=BLUE)
    ax.set_ylim(85, 95)
    ax.set_xticks(M)
    ax2 = ax.twinx()
    ax2.grid(False)
    l2 = ax2.errorbar(M, g[("width_scaled", "mean")], yerr=g[("width_scaled", "std")], fmt="--s", color=RED,
                      lw=LW, ms=2.8, capsize=2, label="Mean interval width")
    ax2.set_ylabel("Mean Interval Width (dB)", color=RED)
    ax2.tick_params(axis="y", colors=RED, direction="in")
    ax.legend(handles=[l1, l2], loc="upper right")
    save(fig, "uq_ensemble_size.png")


def fig_kappa():
    """Gap to the nearest real design and ensemble spread at the optimum, vs kappa (mean ± std)."""
    df = _ms("ms_uq.csv")
    ks = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    tag = lambda k: f"k{k:g}".replace(".", "p")
    gap = np.array([df[f"{tag(k)}_gap"] for k in ks])
    sig = np.array([df[f"{tag(k)}_sigma"] for k in ks])
    fig, (a, b) = plt.subplots(2, 1, figsize=(F.FIGSIZE[0], 3.9), sharex=True, gridspec_kw={"hspace": 0.12})
    a.errorbar(ks, gap.mean(1), yerr=gap.std(1, ddof=1), fmt="-o", color=BLUE, lw=LW, ms=3, capsize=2,
               label="Ensemble, μ + κσ")
    ag = df["aggr_gap"].to_numpy()
    a.axhspan(ag.mean() - ag.std(ddof=1), ag.mean() + ag.std(ddof=1), color=RED, alpha=0.06, lw=0)
    a.axhline(ag.mean(), color=RED, ls="--", lw=LW, label="Single MLP, point optimum")
    a.set_ylabel("Gap to Nearest Real (dB)")
    a.set_ylim(0, None)
    a.legend(loc="upper right")
    a.text(0.02, 0.95, "(a)", transform=a.transAxes, ha="left", va="top")
    b.errorbar(ks, sig.mean(1), yerr=sig.std(1, ddof=1), fmt="-s", color=PURPLE, lw=LW, ms=2.8, capsize=2)
    b.set_xlabel("Risk Weight κ")
    b.set_ylabel("Ensemble Spread σ (dB)")
    b.set_xticks(ks)
    b.set_ylim(0, None)
    b.text(0.02, 0.06, "(b)", transform=b.transAxes, ha="left", va="bottom")
    fig.align_ylabels([a, b])
    save(fig, "uq_kappa.png")


def fig_gen_vs_random():
    df = _ms("ms_generative.csv")
    order = [("Random", "Random\nsampling", "#BFBFBF"), ("GMM", "GMM", BLUE), ("VAE", "VAE", RED),
             ("Diffusion", "Diffusion", GREEN)]
    fig, ax = plt.subplots(figsize=(F.FIGSIZE[0], 2.2))
    ax.grid(axis="x", visible=False)
    x = np.arange(len(order))
    bw = 0.36
    for j, (metric, lab, hatch) in enumerate((("within3", "Within ±3 dB of target", ""),
                                              ("below_mask", "Below −10 dB mask", "////"))):
        for i, (m, _, col) in enumerate(order):
            v = df[df.method == m][metric].to_numpy() * 100
            ax.bar(x[i] + (j - 0.5) * bw, v.mean(), bw * 0.92, yerr=v.std(ddof=1), color=col if j == 0 else "white",
                   edgecolor=col if j else "black", hatch=hatch, lw=0.7, capsize=1.8,
                   error_kw=dict(lw=0.6), label=lab if i == 1 else None, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([o[1] for o in order])
    ax.tick_params(axis="x", which="both", top=False, bottom=False)
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("Share of 400 Designs (%)")
    h = [plt.Rectangle((0, 0), 1, 1, fc="#7F7F7F", ec="black", lw=0.6),
         plt.Rectangle((0, 0), 1, 1, fc="white", ec="black", hatch="////", lw=0.6)]
    ax.legend(h, ["Within ±3 dB of target", "Below −10 dB mask"], loc="upper left", ncol=2)
    save(fig, "generative_vs_random.png")


def fig_guidance_ms():
    df = _ms("ms_guidance.csv")
    g = df.groupby("w")[["below_mask", "diversity"]].agg(["mean", "std"])
    w = g.index.to_numpy()
    fig, ax = plt.subplots(figsize=F.FIGSIZE)
    ax.tick_params(right=False)
    l1 = ax.errorbar(w, g[("below_mask", "mean")] * 100, yerr=g[("below_mask", "std")] * 100, fmt="-o",
                     color=BLUE, lw=LW, ms=3, capsize=2, label="Designs below −10 dB mask")
    ax.set_xlabel("Guidance Scale w")
    ax.set_ylabel("Designs below −10 dB Mask (%)", color=BLUE)
    ax.tick_params(axis="y", colors=BLUE)
    ax.set_xlim(-0.2, 5.2)
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    lo = float((g[("below_mask", "mean")] - g[("below_mask", "std")]).min() * 100)
    ax.set_ylim(min(75, lo - 3), 101)
    ax2 = ax.twinx()
    ax2.grid(False)
    l2 = ax2.errorbar(w, g[("diversity", "mean")], yerr=g[("diversity", "std")], fmt="--s", color=RED, lw=LW,
                      ms=2.8, capsize=2, label="Design diversity")
    ax2.set_ylabel("Design Diversity", color=RED)
    ax2.tick_params(axis="y", colors=RED, direction="in")
    ax2.set_ylim(0.05, 0.30)
    ax.legend(handles=[l1, l2], loc="lower center")
    save(fig, "generative_guidance.png")


def fig_curated_ms():
    df = _ms("ms_curated.csv")
    methods = [("Random", "Random\nSampling"), ("GMM", "GMM"), ("Diffusion+CFG", "Guided\nDiffusion")]
    fig, ax = plt.subplots(figsize=(F.FIGSIZE[0], 2.2))
    ax.grid(axis="x", visible=False)
    x = np.arange(len(methods))
    bw = 0.32
    top = 0
    for j, (data, lab, col) in enumerate((("broad", "Broad training", "#BFBFBF"),
                                          ("curated", "Physics-curated training", GREEN))):
        for i, (m, _) in enumerate(methods):
            v = df[(df.data == data) & (df.method == m)]["below_mask"].to_numpy() * 100
            mu, sd = v.mean(), v.std(ddof=1)
            ax.bar(x[i] + (j - 0.5) * bw, mu, bw, yerr=sd, color=col, edgecolor="black", lw=0.6, capsize=1.8,
                   error_kw=dict(lw=0.6), label=lab if i == 0 else None, zorder=2)
            ax.text(x[i] + (j - 0.5) * bw, mu + sd + 1.5, f"{mu:.1f}", ha="center", va="bottom",
                    fontsize=F.FONT_PT - 1.2, fontweight="bold")
            top = max(top, mu + sd)
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in methods])
    ax.tick_params(axis="x", which="both", top=False, bottom=False)
    ax.set_ylim(0, max(100, top + 22))
    ax.set_ylabel("Designs Meeting −10 dB Mask (%)")
    ax.legend(loc="upper left")
    save(fig, "generative_curated.png")


def fig_cross_ms():
    """Heatmap of mean R2 (± std) over 10 splits from ms_surrogates.csv."""
    from matplotlib.colors import LinearSegmentedColormap
    df = _ms("ms_surrogates.csv")
    sets = sorted(df["dataset"].unique())
    models = ["MLP", "GBDT", "Random Forest", "GP"]
    mu = np.array([[df[(df.dataset == s) & (df.model == m)].R2.mean() for s in sets] for m in models])
    sd = np.array([[df[(df.dataset == s) & (df.model == m)].R2.std(ddof=1) for s in sets] for m in models])
    cols = ["A: CST patch", "B: Rect. patch", "C: TL model"]
    cmap = LinearSegmentedColormap.from_list("rb", ["#FF0000", "#FF8C00", "#FFF2A8", "#7FB3FF", "#0000FF"])
    fig, ax = plt.subplots(figsize=(F.FIGSIZE[0], 2.0))
    ax.grid(False)
    im = ax.imshow(mu, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(mu.shape[0]):
        for j in range(mu.shape[1]):
            v = mu[i, j]
            ink = "white" if (v < 0.22 or v > 0.82) else "black"
            best = i == np.argmax(mu[:, j])
            ax.text(j, i - 0.1, f"{v:.3f}", ha="center", va="center", color=ink,
                    fontsize=F.FONT_PT + 0.6, fontweight="bold" if best else "normal")
            ax.text(j, i + 0.24, f"± {sd[i, j]:.3f}", ha="center", va="center", color=ink,
                    fontsize=F.FONT_PT - 1.4)
    ax.set_xticks(range(len(sets)))
    ax.set_xticklabels(cols)
    ax.set_yticks(range(4))
    ax.set_yticklabels(models)
    ax.tick_params(which="both", length=0, top=False, right=False, bottom=False, left=False)
    ax.set_xticks(np.arange(-0.5, len(sets)), minor=True)
    ax.set_yticks(np.arange(-0.5, 4), minor=True)
    ax.grid(which="minor", color="white", lw=1.2)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label(f"Test R² (mean of {df.seed.nunique()} splits)")
    cb.ax.tick_params(direction="in", length=2)
    cb.outline.set_linewidth(0.6)
    save(fig, "cross_dataset_heatmap.png")


MS_FIGS = {"cov_seeds": fig_cov_seeds, "ens_size": fig_ensemble_size, "kappa": fig_kappa,
           "gen_random": fig_gen_vs_random, "guidance_ms": fig_guidance_ms, "curated_ms": fig_curated_ms,
           "cross_ms": fig_cross_ms}


if __name__ == "__main__":
    F.origin_style()
    picked = [MS_FIGS[a] for a in sys.argv[1:] if a in MS_FIGS]
    if picked:
        for fn in picked:
            fn()
        sys.exit()
    if "curated" in sys.argv:
        fig_curated_simple(); sys.exit()
    if "cross_options" in sys.argv:
        import matplotlib as mpl
        fig_cross_heatmap(); fig_cross_hbar()
        sys.exit()
    only = sys.argv[1:]
    if not only:            # default: exactly the figures used in the paper (multi-seed versions)
        for fn in (fig_spec, fig_uq, fig_gen, *MS_FIGS.values()):
            fn()
        sys.exit()
    # older single-seed drafts, kept for reference: python make_origin_figs.py guidance cross
    for name, fn in (("spec", fig_spec), ("uq", fig_uq), ("gen", fig_gen), ("guidance", fig_guidance),
                     ("cross", fig_cross)):
        if name in only:
            fn()
