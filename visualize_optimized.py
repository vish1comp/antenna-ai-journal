"""Visualize the OPTIMIZED antenna geometry proposed by each optimizer.

After the surrogate-assisted optimization, PSO / GA / Bayesian optimization each return the
geometry they believe best minimizes S11 at 2.4 GHz. This script draws each proposed patch
to scale, plots its surrogate-predicted S11-vs-frequency response, and verifies it against
the nearest real design. It answers: "what does the antenna the model designed look like?"

Run:  python visualize_optimized.py   ->  results/figures/optimized_geometry.png
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import config as C
from data.get_dataset import load_dataset
from models.surrogates import GBDTSurrogate
from optimize.pso import particle_swarm
from optimize.ga import genetic_algorithm
from optimize.bayes_opt import bayesian_optimization
from optimize.surrogate_loop import nearest_true

FR4, COPPER, EDGE, NAVY, BLUE, GRAY = "#cfe3cf", "#c8842e", "#8a5a1e", "#0B3D66", "#1C6FB8", "#586472"


def draw_patch(ax, p, title):
    Ws, Ls = p["width substrate in mm"], p["substrate length in mm"]
    Wp, Lp = p["patch width in mm"], p["patch legth in mm"]
    Lf = p["length impedance line in mm"]; Wf = 3.0
    sw, sl = p["width slot in mm"], p["length slot in mm"]
    Wb = max(Ws, Wp) + 2                      # bounding for a clean draw
    ax.add_patch(Rectangle((-Wb/2, -Ls/2), Wb, Ls, facecolor=FR4, edgecolor="#7fae7f", lw=1.1))
    ax.add_patch(Rectangle((-Wp/2, -Lp/2), Wp, Lp, facecolor=COPPER, edgecolor=EDGE, lw=1.1))
    ax.add_patch(Rectangle((-Wf/2, -Lp/2 - Lf), Wf, Lf, facecolor=COPPER, edgecolor=EDGE, lw=1))
    ax.add_patch(Rectangle((-sl/2, Lp/2 - sw - 1.2), sl, sw, facecolor=FR4, edgecolor=EDGE, lw=0.7))
    pad = 5
    ax.set_xlim(-Wb/2 - pad, Wb/2 + pad); ax.set_ylim(-Ls/2 - Lf - pad, Ls/2 + pad)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, color=NAVY, fontsize=11, fontweight="bold")
    ax.text(0, -Lp/2 + Lp*0.30, f"patch\n{Wp:.1f}×{Lp:.1f}", ha="center", va="center",
            fontsize=8, color="white")
    ax.text(0, -Ls/2 + 1.2, f"$W_s${Ws:.1f} · $L_s${Ls:.1f} · feed {Lf:.1f} · slot {sl:.1f}×{sw:.1f} mm",
            ha="center", va="bottom", fontsize=7.3, color=GRAY)


def sweep(surr, ds, geom, freqs):
    rows = np.vstack([ds["assemble_at"](geom[None, :], f) for f in freqs])
    return surr.predict(rows)


def main():
    os.makedirs(C.FIG_DIR, exist_ok=True)
    ds = load_dataset()
    names = ds["opt_feature_names"]
    surr = GBDTSurrogate(random_state=C.SEED).fit(ds["X"], ds["y"])
    assemble = ds["assemble"]
    tf = ds["target_freq"]

    def obj(P):
        return surr.predict(assemble(P))

    print("Optimizing geometry to minimize S11 at %.2f GHz ..." % tf)
    results = {
        "PSO": particle_swarm(obj, ds["opt_bounds"], minimize=True, seed=C.SEED, **C.PSO),
        "GA": genetic_algorithm(obj, ds["opt_bounds"], minimize=True, seed=C.SEED, **C.GA),
        "Bayesian": bayesian_optimization(obj, ds["opt_bounds"], minimize=True, seed=C.SEED, **C.BO),
    }
    freqs = np.linspace(1.9, 2.9, 140)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), gridspec_kw={"height_ratios": [1.25, 1]})
    table = []
    for j, (opt, res) in enumerate(results.items()):
        x = res["x"]
        p = {n: float(v) for n, v in zip(names, x)}
        pred = float(res["fx"])
        ver = nearest_true(x, ds["verify_X"], ds["verify_y"])
        draw_patch(axes[0, j], p, f"{opt}: designed patch")
        s = sweep(surr, ds, x, freqs)
        axg = axes[1, j]
        axg.plot(freqs, s, color=BLUE, lw=2, label="predicted")
        axg.axhline(-10, color="#B5771F", ls="--", lw=1.1)
        axg.axvline(tf, color=GRAY, lw=0.7, alpha=0.5)
        axg.plot(tf, pred, "o", color=NAVY, ms=6)
        axg.set_title(f"pred S11 = {pred:.1f} dB @ {tf:.2f} GHz", color=NAVY, fontsize=10)
        axg.set_xlabel("frequency (GHz)"); axg.set_ylabel("S11 (dB)")
        axg.grid(True, color="#e6edf4")
        row = {"Optimizer": opt, "pred S11 (dB)": round(pred, 2),
               "nearest-real (dB)": round(ver["y_true"], 2), "evals": res["n_evals"]}
        row.update({n: round(float(v), 2) for n, v in zip(names, x)})
        table.append(row)
        print(f"  {opt:9s} pred S11={pred:6.2f} dB | nearest-real={ver['y_true']:6.2f} dB")

    fig.suptitle("Optimized patch geometry on the Zenodo CST dataset "
                 "(GBDT surrogate, designed by each optimizer)",
                 color=NAVY, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(C.FIG_DIR, "optimized_geometry.png")
    fig.savefig(out, dpi=160); plt.close(fig)

    df = pd.DataFrame(table)
    df.to_csv(os.path.join(C.RESULTS_DIR, "optimized_geometry.csv"), index=False)
    with open(os.path.join(C.RESULTS_DIR, "optimized_geometry.md"), "w", encoding="utf-8") as f:
        f.write("# Optimized geometry per optimizer (surrogate-designed, verified vs nearest real)\n\n"
                + df.to_markdown(index=False) + "\n")
    print("saved", out, "and results/optimized_geometry.csv")


if __name__ == "__main__":
    main()
