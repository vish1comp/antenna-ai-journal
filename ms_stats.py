"""Summaries and paired significance tests for the multi-seed results (results/ms_*.csv).

For every comparison the per-seed differences are tested with a paired t-test when they
pass a Shapiro-Wilk normality check (p > 0.05) and with the Wilcoxon signed-rank test
otherwise; p-values are Holm-corrected within each family of comparisons.

    python ms_stats.py        # prints the tables and writes results/ms_stats.md
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats import shapiro, ttest_rel, wilcoxon

import config as C

R = C.RESULTS_DIR
OUT = []


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    if np.allclose(d, 0):
        return "--", 0.0, 1.0
    use_t = len(d) >= 3 and shapiro(d).pvalue > 0.05
    p = ttest_rel(a, b).pvalue if use_t else wilcoxon(a, b).pvalue
    return ("t-test" if use_t else "Wilcoxon"), float(d.mean()), float(p)


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj = np.empty_like(ps)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def stars(p):
    return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))


def family(title, rows):
    """rows: list of dicts with keys label, a, b (arrays aligned by seed), ref."""
    res = []
    for r in rows:
        test, d, p = paired(r["a"], r["b"])
        base = np.mean(r["b"])
        res.append({**{k: v for k, v in r.items() if k not in ("a", "b")}, "test": test, "delta": d,
                    "delta_pct": 100 * d / abs(base) if base else np.nan, "p": p,
                    "mean_a": np.mean(r["a"]), "mean_b": base, "n": len(r["a"])})
    adj = holm([x["p"] for x in res])
    for x, q in zip(res, adj):
        x["p_holm"] = q
        x["sig"] = stars(q)
    df = pd.DataFrame(res)
    OUT.append(f"## {title}\n\n" + df.to_markdown(index=False, floatfmt=".4g") + "\n")
    print(f"\n=== {title}\n{df.to_string(index=False)}")
    return df


def ms(x):
    return f"{np.mean(x):.3f} ± {np.std(x, ddof=1):.3f}"


def surrogates():
    p = os.path.join(R, "ms_surrogates.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    summ = df.groupby(["dataset", "model"]).agg(R2_mean=("R2", "mean"), R2_std=("R2", "std"),
                                                NRMSE_mean=("NRMSE", "mean"), NRMSE_std=("NRMSE", "std"),
                                                n=("seed", "count")).reset_index()
    OUT.append("## Surrogates (mean ± std over splits)\n\n" + summ.to_markdown(index=False, floatfmt=".4f") + "\n")
    print(summ.to_string(index=False))
    for ds in sorted(df.dataset.unique()):
        d = df[df.dataset == ds].pivot(index="seed", columns="model", values="R2").dropna()
        rows = [{"dataset": ds[:2], "label": f"MLP vs {m}", "a": d["MLP"], "b": d[m]}
                for m in ("GBDT", "Random Forest", "GP")]
        family(f"Surrogate R2, MLP vs baselines, dataset {ds[:2]}", rows)


def uq():
    p = os.path.join(R, "ms_uq.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    keys = ["cov_fixed", "cov_scaled", "cov_raw", "width_fixed", "width_scaled", "width_raw", "spearman", "rmse",
            "aggr_mlp_pred", "aggr_ens_mean", "aggr_sigma", "aggr_sigma_pct", "aggr_real", "aggr_gap"]
    kcols = [c for c in df.columns if c.startswith("k") and c[1].isdigit()]
    lines = [f"| {k} | {ms(df[k])} |" for k in keys + kcols]
    OUT.append(f"## Uncertainty (n = {len(df)} seeds)\n\n| quantity | mean ± std |\n|---|---|\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))
    t = 0.90
    family("Coverage error |cov - 0.90|: conformal vs raw ensemble interval", [
        {"label": "fixed conformal vs raw", "a": (df.cov_fixed - t).abs(), "b": (df.cov_raw - t).abs()},
        {"label": "scaled conformal vs raw", "a": (df.cov_scaled - t).abs(), "b": (df.cov_raw - t).abs()}])
    family("Gap to nearest real design (dB)", [
        {"label": "risk-averse k=1.5 vs single MLP", "a": df.k1p5_gap, "b": df.aggr_gap},
        {"label": "risk-averse k=1.5 vs ensemble mean k=0", "a": df.k1p5_gap, "b": df.k0_gap}])


def generative():
    p = os.path.join(R, "ms_generative.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        summ = df.groupby("method")[["median", "within3", "below_mask", "diversity", "secs"]].agg(["mean", "std"])
        OUT.append("## Generative (mean ± std over seeds)\n\n" + summ.to_markdown(floatfmt=".3f") + "\n")
        print(summ.to_string())
        for metric in ("within3", "below_mask"):
            d = df.pivot(index="seed", columns="method", values=metric).dropna()
            have = [m for m in ("GMM", "VAE", "Diffusion") if m in d.columns]
            if have and len(d) >= 3:
                family(f"Generators vs random sampling, {metric}",
                       [{"label": f"{m} vs Random", "a": d[m], "b": d["Random"]} for m in have])
    p = os.path.join(R, "ms_guidance.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        summ = df.groupby("w")[["median", "within3", "below_mask", "diversity"]].agg(["mean", "std"])
        OUT.append("## Guidance (mean ± std over seeds)\n\n" + summ.to_markdown(floatfmt=".3f") + "\n")
        print(summ.to_string())
        rows = []
        for metric in ("below_mask", "diversity"):
            d = df.pivot(index="seed", columns="w", values=metric).dropna()
            for w in (2.0, 3.0, 5.0):
                rows.append({"label": f"{metric}: w={w:g} vs w=1", "a": d[w], "b": d[1.0]})
        family("Guidance scale vs w = 1", rows)
    p = os.path.join(R, "ms_curated.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        summ = df.groupby(["data", "method"])[["median", "below_mask", "diversity"]].agg(["mean", "std"])
        OUT.append("## Curated (mean ± std over seeds)\n\n" + summ.to_markdown(floatfmt=".3f") + "\n")
        print(summ.to_string())
        rows = []
        for m in ("GMM", "Diffusion+CFG", "Random"):
            d = df[df.method == m].pivot(index="seed", columns="data", values="below_mask").dropna()
            rows.append({"label": f"{m}: curated vs broad", "a": d["curated"], "b": d["broad"]})
        family("Curated vs broad training, below_mask (exact physics)", rows)


if __name__ == "__main__":
    surrogates()
    uq()
    generative()
    with open(os.path.join(R, "ms_stats.md"), "w", encoding="utf-8") as f:
        f.write("# Multi-seed results and paired tests\n\n" + "\n".join(OUT))
    print("\nwrote", os.path.join(R, "ms_stats.md"))
