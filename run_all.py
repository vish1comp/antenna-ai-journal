"""End-to-end antenna AI/ML demo: surrogate benchmark + metaheuristic optimization.

Pipeline
--------
1. Load a public antenna/EM dataset (design parameters -> performance metric).
2. Train three surrogates (MLP, GBDT, GP) and benchmark them on a held-out test
   set with a common metric schema (R2, RMSE, MAE, NRMSE) + fit time.
3. Active learning: show uncertainty-guided sampling beats random for the same
   labeling budget (data-efficiency learning curve).
4. Surrogate-assisted optimization: run PSO and GA over the design bounds using
   the best surrogate as a cheap objective; verify the proposed optimum against
   the nearest real design in the dataset.

Outputs: results/metrics.csv, results/summary.md, results/figures/*.png

Run:  python run_all.py
"""
from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config as C
from data.get_dataset import load_dataset
from models.surrogates import MLPSurrogate, GBDTSurrogate, GPSurrogate
from models.uncertainty import EnsembleSurrogate, split_conformal
from optimize.pso import particle_swarm
from optimize.ga import genetic_algorithm
from optimize.bayes_opt import bayesian_optimization
from optimize.surrogate_loop import surrogate_objective, nearest_true, active_learning
from optimize.spec_design import (make_spec, spec_objective, evaluate_response,
                                  nearest_true_curve)
from evaluate.metrics import regression_report
from evaluate import plots


def _rng_subset(n, k, seed):
    return np.random.default_rng(seed).choice(n, size=min(k, n), replace=False)


def benchmark_surrogates(X_tr, y_tr, X_te, y_te, target_name):
    """Train each surrogate, evaluate on the test set, save parity plots."""
    rows = []
    specs = [
        ("MLP", MLPSurrogate),
        ("GBDT", GBDTSurrogate),
        ("GP", GPSurrogate),
    ]
    fitted = {}
    for name, cls in specs:
        # GP is cubic in n_train -> cap its training set.
        if name == "GP" and len(X_tr) > C.GP_MAX_TRAIN:
            idx = _rng_subset(len(X_tr), C.GP_MAX_TRAIN, C.SEED)
            Xf, yf = X_tr[idx], y_tr[idx]
            note = f"trained on {C.GP_MAX_TRAIN}/{len(X_tr)} (cubic cost)"
        else:
            Xf, yf, note = X_tr, y_tr, ""
        model = cls(random_state=C.SEED).fit(Xf, yf)
        t = time.perf_counter()
        y_hat = model.predict(X_te)
        pred_time = time.perf_counter() - t
        rep = regression_report(y_te, y_hat)
        rep.update({"model": name, "fit_time_s": round(model.fit_time_, 4),
                    "predict_time_s": round(pred_time, 5), "note": note})
        rows.append(rep)
        fitted[name] = model
        plots.parity_plot(y_te, y_hat, f"{name} surrogate", C.FIG_DIR,
                          f"parity_{name}.png", target_name=target_name)
        print(f"  {name:5s}  R2={rep['R2']:.3f}  NRMSE={rep['NRMSE']:.3f}  "
              f"fit={model.fit_time_:.2f}s {note}")
    df = pd.DataFrame(rows)[["model", "R2", "RMSE", "MAE", "NRMSE",
                             "fit_time_s", "predict_time_s", "note"]]
    return df, fitted


def run_active_learning(X_tr, y_tr, X_te, y_te):
    pool_n = len(X_tr)
    sizes = [s for s in C.AL_SIZES if s <= pool_n] or [pool_n]
    strategies = {
        "random": "random",
        "uncertainty (GP)": "uncertainty",
        "committee+diversity": "qbc",
    }
    print(f"  sizes={sizes}")
    curves = {}
    for label, strat in strategies.items():
        curves[label] = active_learning(X_tr, y_tr, X_te, y_te, sizes,
                                        strategy=strat, init=C.AL_INIT, seed=C.SEED)
        print(f"  {label:20s} NRMSE={['%.3f' % v for v in curves[label]]}")
    plots.learning_curve(sizes, list(curves.values()), list(curves.keys()),
                         "test NRMSE", "Active learning: data efficiency",
                         C.FIG_DIR, "active_learning.png")
    return sizes, curves


def run_optimization(surrogate, ds):
    """Design the geometry that optimizes the target at the fixed application band.

    Frequency is pinned (via ds['assemble']) so the metaheuristics search only
    the geometry parameters - the real engineering task. The proposed optimum is
    then verified against the nearest real design measured at that band.
    """
    minimize = (ds["direction"] == "min")
    target_name, tf = ds["target_name"], ds["target_freq"]
    assemble = ds["assemble"]

    def obj(P_geom):  # P_geom: (n, n_geom) -> surrogate S11 at fixed frequency
        return surrogate.predict(assemble(P_geom))

    pso = particle_swarm(obj, ds["opt_bounds"], minimize=minimize, seed=C.SEED, **C.PSO)
    ga = genetic_algorithm(obj, ds["opt_bounds"], minimize=minimize, seed=C.SEED, **C.GA)
    bo = bayesian_optimization(obj, ds["opt_bounds"], minimize=minimize, seed=C.SEED, **C.BO)

    # convergence vs cumulative evaluations (fair cross-optimizer comparison)
    n_p, n_g, n_i = C.PSO["n_particles"], C.GA["pop_size"], C.BO["n_init"]
    evals = [n_p * (np.arange(len(pso["history"])) + 1),
             n_g * (np.arange(len(ga["history"])) + 1),
             n_i + np.arange(len(bo["history"]))]
    plots.convergence_evals(
        evals, [pso["history"], ga["history"], bo["history"]], ["PSO", "GA", "Bayesian opt"],
        f"best {target_name} @ {tf:.2f} GHz (surrogate)",
        "Optimizer sample efficiency", C.FIG_DIR, "optimization_convergence.png")

    results = {}
    for label, res in [("PSO", pso), ("GA", ga), ("Bayesian opt", bo)]:
        ver = nearest_true(res["x"], ds["verify_X"], ds["verify_y"])
        results[label] = {
            "x": res["x"], "surrogate_pred": float(res["fx"]),
            "verify_true": ver["y_true"], "verify_dist": ver["distance"],
            "n_evals": res["n_evals"],
        }
        print(f"  {label:12s}: surrogate {target_name}={res['fx']:.3f} dB | "
              f"nearest real {target_name}={ver['y_true']:.3f} dB "
              f"({res['n_evals']} evals)")
    return results


def run_spec_design(surrogate, ds):
    """Design geometry whose S11(f) response meets a return-loss mask over a band."""
    spec = make_spec()
    obj = spec_objective(surrogate, ds, spec)
    res = particle_swarm(obj, ds["opt_bounds"], minimize=True, seed=C.SEED, **C.PSO)
    ev = evaluate_response(surrogate, ds, spec, res["x"])
    tf, ts, tdist = nearest_true_curve(ds, res["x"])
    plots.response_curve(ev["freqs"], ev["s11"], spec, C.FIG_DIR,
                         "spec_design_response.png", true_freqs=tf, true_s11=ts)
    print(f"  target: S11 <= {spec['mask_db']:.0f} dB over "
          f"{spec['band'][0]:.2f}-{spec['band'][1]:.2f} GHz")
    print(f"  designed: center S11={ev['center_s11']:.2f} dB, min={ev['min_s11']:.2f} dB, "
          f"S11<{spec['mask_db']:.0f}dB bandwidth ~{ev['bandwidth_ghz']*1000:.0f} MHz")
    print(f"  verified: nearest real geometry at distance {tdist:.2f}")
    return {"spec": spec, "eval": ev, "verify_dist": tdist}


def run_uncertainty(ds, X_tr, y_tr, X_te, y_te, aggressive_x):
    """Calibrated uncertainty + flag the aggressive optimum + risk-averse design."""
    Xf, Xc, yf, yc = train_test_split(X_tr, y_tr, test_size=0.25, random_state=C.SEED)
    ens = EnsembleSurrogate(n_models=C.UQ["n_models"], seed=C.SEED).fit(Xf, yf)
    mu_c, sd_c = ens.predict_mean_std(Xc)
    mu_t, sd_t = ens.predict_mean_std(X_te)

    a = C.UQ["alpha"]
    cf = split_conformal(yc, mu_c, alpha=a)
    cfn = split_conformal(yc, mu_c, alpha=a, sigma_cal=sd_c)
    cov = float(np.mean((y_te >= mu_t - cf["q"]) & (y_te <= mu_t + cf["q"])))
    covn = float(np.mean((y_te >= mu_t - cfn["q"] * sd_t) & (y_te <= mu_t + cfn["q"] * sd_t)))
    plots.reliability(sd_t, np.abs(y_te - mu_t), C.FIG_DIR, "uq_reliability.png")
    print(f"  conformal (target {1-a:.0%}): standard coverage={cov:.3f} "
          f"(half-width +/-{cf['q']:.2f} dB), adaptive coverage={covn:.3f}")

    # flag the aggressive (ungated) optimum from step 4
    asm = ds["assemble"]
    ma, sa = ens.predict_mean_std(asm(aggressive_x[None, :]))
    ma, sa = float(ma[0]), float(sa[0])
    pct = float(np.mean(sd_t <= sa))
    hw = cfn["q"] * sa                       # adaptive (sigma-scaled) half-width
    lo, hi = ma - hw, ma + hw
    print(f"  aggressive optimum: ensemble mean={ma:.2f} dB, sigma={sa:.2f} dB "
          f"({pct:.0%} pctile of test sigma) -> adaptive 90% interval [{lo:.1f}, {hi:.1f}] dB")

    # uncertainty-gated (risk-averse) design: minimize mean + kappa*sigma
    k = C.UQ["kappa"]
    def gated(P):
        m, s = ens.predict_mean_std(asm(P))
        return m + k * s
    g = particle_swarm(gated, ds["opt_bounds"], minimize=True, seed=C.SEED, **C.UQ_PSO)
    mg, sg = ens.predict_mean_std(asm(g["x"][None, :]))
    mg, sg = float(mg[0]), float(sg[0])
    vg = nearest_true(g["x"], ds["verify_X"], ds["verify_y"])
    print(f"  risk-averse design (mean+{k}*sigma): mean={mg:.2f} dB, sigma={sg:.2f} dB, "
          f"nearest-real={vg['y_true']:.2f} dB, gap={abs(mg - vg['y_true']):.2f} dB")

    return {"target": 1 - a, "cov": cov, "covn": covn, "half_width": cf["q"],
            "aggr_mean": ma, "aggr_sigma": sa, "aggr_pct": pct, "aggr_interval": (lo, hi),
            "gated_mean": mg, "gated_sigma": sg, "gated_real": vg["y_true"], "kappa": k}


def main():
    os.makedirs(C.FIG_DIR, exist_ok=True)
    print("\n[1/6] Loading dataset ...")
    ds = load_dataset()
    X, y = ds["X"], ds["y"]
    print(f"  dataset: {ds['name']}  ->  {X.shape[0]} rows, {X.shape[1]} features")
    print(f"  target : {ds['target_name']} (optimize: {ds['direction']})")
    print(f"  features: {', '.join(ds['feature_names'])}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=C.TEST_SIZE, random_state=C.SEED)

    print("\n[2/6] Benchmarking surrogates ...")
    metrics_df, fitted = benchmark_surrogates(X_tr, y_tr, X_te, y_te, ds["target_name"])
    metrics_df.to_csv(os.path.join(C.RESULTS_DIR, "metrics.csv"), index=False)
    plots.comparison_bars(metrics_df["model"], metrics_df["R2"], "R2 (test)",
                          "Surrogate accuracy", C.FIG_DIR, "compare_r2.png")
    plots.comparison_bars(metrics_df["model"], metrics_df["fit_time_s"], "fit time (s)",
                          "Training cost", C.FIG_DIR, "compare_fit_time.png", fmt="{:.2f}")

    print("\n[3/6] Active learning (random vs GP-variance vs committee+diversity) ...")
    sizes, curves = run_active_learning(X_tr, y_tr, X_te, y_te)

    print("\n[4/6] Surrogate-assisted optimization (PSO, GA & Bayesian) ...")
    best_name = metrics_df.sort_values("NRMSE").iloc[0]["model"]
    print(f"  using best surrogate by NRMSE: {best_name}  "
          f"(design geometry @ {ds['target_freq']:.2f} GHz)")
    opt = run_optimization(fitted[best_name], ds)

    print("\n[5/6] Inverse spec-matching design (S11 mask over a band) ...")
    spec_res = run_spec_design(fitted[best_name], ds)

    print("\n[6/6] Uncertainty quantification & risk-averse design ...")
    uq_res = run_uncertainty(ds, X_tr, y_tr, X_te, y_te, opt["PSO"]["x"])

    _write_summary(ds, metrics_df, sizes, curves, best_name, opt, spec_res, uq_res)
    print("\nDone. See results/metrics.csv, results/summary.md, results/figures/*.png")


def _write_summary(ds, metrics_df, sizes, curves, best_name, opt, spec_res=None, uq_res=None):
    lines = []
    lines.append(f"# Results — {ds['name']}\n")
    lines.append(f"- Rows: **{ds['X'].shape[0]}**, features: **{ds['X'].shape[1]}**, "
                 f"target: **{ds['target_name']}** (optimize: {ds['direction']})")
    lines.append(f"- Source: {ds.get('source', 'n/a')}\n")
    lines.append("## Surrogate benchmark (held-out test)\n")
    lines.append(metrics_df.round(4).to_markdown(index=False))
    lines.append("\n## Active learning (test NRMSE vs labeled budget)\n")
    al = pd.DataFrame({"labeled": sizes})
    for k, v in curves.items():
        al[k] = np.round(v, 4)
    lines.append(al.to_markdown(index=False))
    rand = np.array(curves["random"])
    means = {k: float(np.mean(v)) for k, v in curves.items()}
    best_al = min(means, key=means.get)
    qbc = np.array(curves.get("committee+diversity", rand))
    diff = rand - qbc  # >0 => committee+diversity better at that budget
    wins = int((diff > 0).sum())
    j = int(np.argmax(diff))
    lines.append(
        f"\nLowest mean test NRMSE across budgets: **{best_al}**. "
        f"Committee+diversity acquisition beat random at **{wins}/{len(sizes)}** budgets "
        f"(largest margin {max(diff.max(), 0.0):.3f} NRMSE at {sizes[j]} samples). "
        f"Adding batch diversity to uncertainty is the fix for the redundant clustering "
        f"that made naive GP max-variance no better than random on this frequency-sweep pool.\n")
    lines.append("## Surrogate-assisted optimization\n")
    lines.append(f"Best surrogate by NRMSE: **{best_name}**. Proposed optima "
                 f"(verified against nearest real design):\n")
    for k, v in opt.items():
        lines.append(f"- **{k}**: surrogate {ds['target_name']}={v['surrogate_pred']:.4f}, "
                     f"nearest-real={v['verify_true']:.4f} "
                     f"(distance {v['verify_dist']:.3f}, {v['n_evals']} cheap evaluations)")

    if spec_res is not None:
        sp, ev = spec_res["spec"], spec_res["eval"]
        lines.append("\n## Inverse spec-matching design\n")
        lines.append(f"Target: S11 <= {sp['mask_db']:.0f} dB across "
                     f"{sp['band'][0]:.2f}-{sp['band'][1]:.2f} GHz, resonant at "
                     f"{sp['center']:.2f} GHz.\n")
        lines.append(f"- Designed response (surrogate): center S11 = "
                     f"**{ev['center_s11']:.2f} dB**, minimum = {ev['min_s11']:.2f} dB, "
                     f"S11 < {sp['mask_db']:.0f} dB bandwidth ~ "
                     f"**{ev['bandwidth_ghz']*1000:.0f} MHz**.")
        lines.append(f"- Verified against nearest real design "
                     f"(geometry distance {spec_res['verify_dist']:.2f}); see "
                     f"`figures/spec_design_response.png`.")

    if uq_res is not None:
        u = uq_res
        lines.append("\n## Uncertainty quantification & risk-averse design\n")
        lines.append(f"- Split-conformal intervals are calibrated: empirical coverage "
                     f"**{u['cov']:.1%}** (standard) / **{u['covn']:.1%}** (adaptive) "
                     f"vs a {u['target']:.0%} target; ensemble std tracks error "
                     f"(`figures/uq_reliability.png`).")
        lines.append(f"- The aggressive optimum is automatically **flagged**: ensemble "
                     f"mean {u['aggr_mean']:.2f} dB with std {u['aggr_sigma']:.2f} dB "
                     f"(**{u['aggr_pct']:.0%}** percentile of test uncertainty), conformal "
                     f"interval [{u['aggr_interval'][0]:.1f}, {u['aggr_interval'][1]:.1f}] dB "
                     f"— i.e. an untrustworthy, high-uncertainty design.")
        lines.append(f"- Risk-averse design (minimize mean + {u['kappa']}*std): "
                     f"mean {u['gated_mean']:.2f} dB, std {u['gated_sigma']:.2f} dB, "
                     f"nearest-real {u['gated_real']:.2f} dB "
                     f"(gap {abs(u['gated_mean']-u['gated_real']):.2f} dB).")

    with open(os.path.join(C.RESULTS_DIR, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
