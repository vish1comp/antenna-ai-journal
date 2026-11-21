"""Re-run generate_guidance.py's sweep with the paper's seed and save the per-scale
metrics to results/guidance.npz, so Fig. 6 can be redrawn from exact values.

    python regen_guidance_data.py
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

import config as C
import generate_guidance as G


def main():
    ds = G.load_dataset()
    geom, fcol, tcol = ds["geom_cols"], ds["freq_col"], ds["target_col"]
    fwd = G.GBDTSurrogate(random_state=C.SEED).fit(ds["X"], ds["y"])
    tf = ds["target_freq"]
    lb = np.array([b[0] for b in ds["opt_bounds"]])
    ub = np.array([b[1] for b in ds["opt_bounds"]])
    Xg = ds["df"][geom].to_numpy(float)
    Cc = ds["df"][[fcol, tcol]].to_numpy(float)
    target = np.array([tf, G.TARGET_S11])
    diff = G.DiffusionInverseDesigner(T=200, epochs=150, batch=1024, p_uncond=0.15, seed=C.SEED).fit(Xg, Cc, verbose=False)
    out = {"w": [], "median": [], "within3": [], "below_mask": [], "diversity": []}
    for w in G.SCALES:
        samp = np.clip(diff.sample(target, n=G.N_GEN, guidance_scale=w, seed=C.SEED), lb, ub)
        a = fwd.predict(ds["assemble"](samp))
        out["w"].append(w)
        out["median"].append(float(np.median(a)))
        out["within3"].append(float(np.mean(np.abs(a - G.TARGET_S11) <= 3.0)))
        out["below_mask"].append(float(np.mean(a <= G.MASK)))
        out["diversity"].append(float(np.mean(samp.std(0) / (ub - lb))))
        print(f"w={w:.0f}  median {out['median'][-1]:.1f}  within3 {out['within3'][-1]:.1%}  "
              f"below {out['below_mask'][-1]:.1%}  diversity {out['diversity'][-1]:.3f}", flush=True)
    path = os.path.join(C.RESULTS_DIR, "guidance.npz")
    np.savez(path, **{k: np.array(v) for k, v in out.items()})
    print("wrote", path)


if __name__ == "__main__":
    main()
