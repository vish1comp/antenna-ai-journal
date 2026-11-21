"""Classifier-free guidance: a tunable fidelity <-> diversity knob for diffusion.

Trains one conditional diffusion model with condition dropout, then samples the
same spec at several guidance scales w. Increasing w sharpens adherence to the
target (higher spec fidelity) at the cost of design diversity - the standard fix
for the fidelity gap plain conditional diffusion showed in generate_compare.py.

Run:  python generate_guidance.py     (needs torch; CPU is fine)
Outputs: results/generative_guidance.md, results/figures/generative_guidance.png
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

import config as C
from data.get_dataset import load_dataset
from models.surrogates import GBDTSurrogate
from generate.diffusion import DiffusionInverseDesigner
from evaluate import plots

TARGET_S11 = -20.0
MASK = -10.0
N_GEN = 400
SCALES = [0.0, 1.0, 2.0, 3.0, 5.0]


def main():
    os.makedirs(C.FIG_DIR, exist_ok=True)
    print("[1/3] Loading data + forward surrogate ...")
    ds = load_dataset()
    geom, fcol, tcol = ds["geom_cols"], ds["freq_col"], ds["target_col"]
    fwd = GBDTSurrogate(random_state=C.SEED).fit(ds["X"], ds["y"])
    tf = ds["target_freq"]
    lb = np.array([b[0] for b in ds["opt_bounds"]])
    ub = np.array([b[1] for b in ds["opt_bounds"]])
    Xg = ds["df"][geom].to_numpy(float)
    Cc = ds["df"][[fcol, tcol]].to_numpy(float)
    target = np.array([tf, TARGET_S11])

    print("[2/3] Training CFG diffusion (condition dropout p=0.15) ...")
    diff = DiffusionInverseDesigner(T=200, epochs=150, batch=1024,
                                    p_uncond=0.15, seed=C.SEED).fit(Xg, Cc, verbose=True)

    print(f"[3/3] Sampling at guidance scales {SCALES} (spec S11={TARGET_S11:.0f} dB @ {tf:.2f} GHz):")
    rows = []
    for w in SCALES:
        samp = np.clip(diff.sample(target, n=N_GEN, guidance_scale=w, seed=C.SEED), lb, ub)
        achieved = fwd.predict(ds["assemble"](samp))
        rows.append({
            "w": w,
            "median": float(np.median(achieved)),
            "within3": float(np.mean(np.abs(achieved - TARGET_S11) <= 3.0)),
            "below_mask": float(np.mean(achieved <= MASK)),
            "diversity": float(np.mean(samp.std(0) / (ub - lb))),
        })
        print(f"    w={w:>3.0f}  median={rows[-1]['median']:>6.1f} dB  "
              f"within3={rows[-1]['within3']:>4.0%}  below_mask={rows[-1]['below_mask']:>4.0%}  "
              f"diversity={rows[-1]['diversity']:.2f}")

    ws = [r["w"] for r in rows]
    plots.guidance_curve(ws, [r["below_mask"] for r in rows], [r["diversity"] for r in rows],
                         C.FIG_DIR, "generative_guidance.png")
    _summary(tf, rows)
    print("\nDone. See results/generative_guidance.md and figures/generative_guidance.png")


def _summary(tf, rows):
    lines = [
        "# Classifier-free guidance — diffusion fidelity vs diversity knob\n",
        f"One CFG diffusion model (condition dropout p=0.15), sampled for "
        f"**S11 = {TARGET_S11:.0f} dB @ {tf:.2f} GHz** at several guidance scales w.\n",
        "| Guidance w | Median S11 (dB) | Within 3 dB | Below −10 dB mask | Diversity |",
        "|---|---|---|---|---|",
        *[f"| {r['w']:.0f} | {r['median']:.1f} | {r['within3']:.0%} | "
          f"{r['below_mask']:.0%} | {r['diversity']:.2f} |" for r in rows],
        "",
        "w = 0 is the unconditional model (ignores the spec); w = 1 is the plain "
        "conditional model; w > 1 amplifies conditioning. As w increases, spec "
        "fidelity rises and diversity falls — the tunable knob that lets a designer "
        "trade portfolio breadth for tighter spec adherence, closing the fidelity "
        "gap plain conditional diffusion showed against the GMM baseline.",
    ]
    with open(os.path.join(C.RESULTS_DIR, "generative_guidance.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
