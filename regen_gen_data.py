"""Re-run generate_compare.py's three generators with the paper's seed and save
each model's 400 surrogate-scored S11 values to results/gen_compare.npz, so
Fig. 5 can be redrawn from the real data.

    python regen_gen_data.py
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

import config as C
import generate_compare as G


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

    Z = ds["df"][geom + [fcol, tcol]].to_numpy(float)
    gmm = G.ConditionalGMM(n_components=40, seed=C.SEED).fit(Z)
    g = G.metrics(gmm.sample([len(geom), len(geom) + 1], target, n=G.N_GEN, seed=C.SEED)[0], ds, fwd, lb, ub)
    print("GMM done", flush=True)
    cvae = G.CVAEInverseDesigner(epochs=120, batch=1024, beta=0.5, seed=C.SEED).fit(Xg, Cc, verbose=False)
    v = G.metrics(cvae.sample(target, n=G.N_GEN, seed=C.SEED), ds, fwd, lb, ub)
    print("VAE done", flush=True)
    diff = G.DiffusionInverseDesigner(T=200, epochs=150, batch=1024, seed=C.SEED).fit(Xg, Cc, verbose=False)
    d = G.metrics(diff.sample(target, n=G.N_GEN, seed=C.SEED), ds, fwd, lb, ub)
    print("diffusion done", flush=True)

    for name, m in (("GMM", g), ("VAE", v), ("Diffusion", d)):
        print(f"{name:10s} median {m['median']:.1f}  within3 {m['within3']:.0%}  "
              f"below_mask {m['below_mask']:.0%}  diversity {m['diversity']:.2f}  modes {m['modes']}")
    out = os.path.join(C.RESULTS_DIR, "gen_compare.npz")
    np.savez(out, gmm=g["achieved"], vae=v["achieved"], diffusion=d["achieved"],
             div=np.array([g["diversity"], v["diversity"], d["diversity"]]),
             modes=np.array([g["modes"], v["modes"], d["modes"]]),
             target=G.TARGET_S11, mask=G.MASK, freq=tf)
    print("wrote", out)


if __name__ == "__main__":
    main()
