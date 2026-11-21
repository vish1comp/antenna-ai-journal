"""Multi-seed experiments with paired significance tests (journal revision).

Each phase repeats an experiment of the paper over several random seeds, adds a
non-learning baseline where one is meaningful, and writes one CSV row per seed
to results/ms_*.csv as it goes, so a partial run still leaves usable data.

  python exp_multiseed.py surrogates   # 4 surrogates x 3 datasets x 10 splits
  python exp_multiseed.py uq           # conformal vs raw-ensemble intervals, kappa sweep,
                                       #   single-MLP aggressive optimum, 10 seeds
  python exp_multiseed.py ensemble     # ensemble-size sensitivity M = 2..12, 3 seeds
  python exp_multiseed.py generative   # GMM / VAE / diffusion / random baseline, 5 seeds
  python exp_multiseed.py guidance     # CFG scale sweep, 5 seeds
  python exp_multiseed.py curated      # broad vs curated training on dataset C, 5 seeds

Statistics (paired over seeds) are computed afterwards by ms_stats.py.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

import config as C

warnings.filterwarnings("ignore")
SEEDS10 = [42, 7, 123, 2024, 11, 99, 314, 1000, 5, 77]
SEEDS5 = SEEDS10[:5]
KAPPAS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]


def append(name, row):
    path = os.path.join(C.RESULTS_DIR, name)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def done(name, **keys):
    """True if a row with these key values is already in the CSV (lets a run resume)."""
    path = os.path.join(C.RESULTS_DIR, name)
    if not os.path.exists(path):
        return False
    df = pd.read_csv(path)
    m = np.ones(len(df), bool)
    for k, v in keys.items():
        m &= df[k].astype(str).to_numpy() == str(v)
    return bool(m.any())


# ------------------------------------------------------------------ surrogates
def phase_surrogates():
    import benchmark_datasets as B
    from evaluate.metrics import regression_report
    for loader in (B.load_zenodo, B.load_hf, B.gen_synthetic):
        X, y, name = loader()
        X, y = B._subsample(X, y, B.N_MAX)
        for seed in SEEDS10:
            if done("ms_surrogates.csv", dataset=name, seed=seed):
                continue
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed)
            for m in B.MODELS:
                t = time.perf_counter()
                mdl = B.make(m, seed)
                if m == "GP" and len(Xtr) > C.GP_MAX_TRAIN:
                    idx = np.random.default_rng(seed).choice(len(Xtr), C.GP_MAX_TRAIN, replace=False)
                    mdl.fit(Xtr[idx], ytr[idx])
                else:
                    mdl.fit(Xtr, ytr)
                rep = regression_report(yte, mdl.predict(Xte))
                append("ms_surrogates.csv", {"dataset": name, "seed": seed, "model": m,
                                             "R2": rep["R2"], "NRMSE": rep["NRMSE"],
                                             "fit_s": time.perf_counter() - t})
            print(f"surrogates {name[:2]} seed {seed} done", flush=True)


# ------------------------------------------------------------------ uncertainty + kappa
def phase_uq():
    from run_all import load_dataset
    from models.uncertainty import EnsembleSurrogate, split_conformal
    from models.surrogates import MLPSurrogate
    from optimize.pso import particle_swarm
    from optimize.surrogate_loop import nearest_true
    ds = load_dataset()
    asm = ds["assemble"]
    a = C.UQ["alpha"]
    z = 1.6448536269514722                         # raw Gaussian 90% interval
    for seed in SEEDS10:
        if done("ms_uq.csv", seed=seed):
            continue
        t0 = time.perf_counter()
        X_tr, X_te, y_tr, y_te = train_test_split(ds["X"], ds["y"], test_size=C.TEST_SIZE, random_state=seed)
        Xf, Xc, yf, yc = train_test_split(X_tr, y_tr, test_size=0.25, random_state=seed)
        ens = EnsembleSurrogate(n_models=C.UQ["n_models"], seed=seed).fit(Xf, yf)
        mu_c, sd_c = ens.predict_mean_std(Xc)
        mu_t, sd_t = ens.predict_mean_std(X_te)
        err = np.abs(y_te - mu_t)
        cf = split_conformal(yc, mu_c, alpha=a)
        cfn = split_conformal(yc, mu_c, alpha=a, sigma_cal=sd_c)
        row = {"seed": seed,
               "cov_fixed": float(np.mean(err <= cf["q"])),
               "cov_scaled": float(np.mean(err <= cfn["q"] * sd_t)),
               "cov_raw": float(np.mean(err <= z * sd_t)),            # uncalibrated ensemble
               "width_fixed": 2 * cf["q"],
               "width_scaled": float(np.mean(2 * cfn["q"] * sd_t)),
               "width_raw": float(np.mean(2 * z * sd_t)),
               "spearman": float(spearmanr(sd_t, err).correlation),
               "rmse": float(np.sqrt(np.mean(err ** 2)))}

        # single-network (MLP) aggressive optimum, judged by the ensemble and the data
        mlp = MLPSurrogate(random_state=seed).fit(X_tr, y_tr)
        g = particle_swarm(lambda P: mlp.predict(asm(P)), ds["opt_bounds"], minimize=True, seed=seed, **C.PSO)
        ma, sa = ens.predict_mean_std(asm(g["x"][None, :]))
        v = nearest_true(g["x"], ds["verify_X"], ds["verify_y"])
        row.update({"aggr_mlp_pred": float(g["fx"]), "aggr_ens_mean": float(ma[0]),
                    "aggr_sigma": float(sa[0]), "aggr_sigma_pct": float(np.mean(sd_t <= sa[0])),
                    "aggr_hw": float(cfn["q"] * sa[0]), "aggr_real": v["y_true"],
                    "aggr_gap": abs(float(g["fx"]) - v["y_true"])})

        # kappa sweep on the ensemble objective mu + kappa * sigma
        for k in KAPPAS:
            def obj(P, k=k):
                m, s = ens.predict_mean_std(asm(P))
                return m + k * s
            r = particle_swarm(obj, ds["opt_bounds"], minimize=True, seed=seed, **C.UQ_PSO)
            m, s = ens.predict_mean_std(asm(r["x"][None, :]))
            vk = nearest_true(r["x"], ds["verify_X"], ds["verify_y"])
            tag = f"k{k:g}".replace(".", "p")
            row.update({f"{tag}_mean": float(m[0]), f"{tag}_sigma": float(s[0]),
                        f"{tag}_real": vk["y_true"], f"{tag}_gap": abs(float(m[0]) - vk["y_true"]),
                        f"{tag}_dist": vk["distance"]})
        row["secs"] = time.perf_counter() - t0
        append("ms_uq.csv", row)
        print(f"uq seed {seed}: cov {row['cov_fixed']:.3f}/{row['cov_scaled']:.3f} raw {row['cov_raw']:.3f} "
              f"| aggr gap {row['aggr_gap']:.1f} | k1.5 gap {row['k1p5_gap']:.1f} | {row['secs']:.0f}s",
              flush=True)


def phase_ensemble():
    from run_all import load_dataset
    from models.uncertainty import EnsembleSurrogate, split_conformal
    ds = load_dataset()
    a = C.UQ["alpha"]
    for seed in SEEDS10[:3]:
        if done("ms_ensemble.csv", seed=seed):
            continue
        X_tr, X_te, y_tr, y_te = train_test_split(ds["X"], ds["y"], test_size=C.TEST_SIZE, random_state=seed)
        Xf, Xc, yf, yc = train_test_split(X_tr, y_tr, test_size=0.25, random_state=seed)
        ens = EnsembleSurrogate(n_models=12, seed=seed).fit(Xf, yf)
        Sc, St = ens._stack(Xc), ens._stack(X_te)
        for M in (2, 4, 6, 8, 10, 12):
            mu_c, sd_c = Sc[:M].mean(0), Sc[:M].std(0)
            mu_t, sd_t = St[:M].mean(0), St[:M].std(0)
            err = np.abs(y_te - mu_t)
            cfn = split_conformal(yc, mu_c, alpha=a, sigma_cal=sd_c)
            append("ms_ensemble.csv", {"seed": seed, "M": M,
                                        "cov_scaled": float(np.mean(err <= cfn["q"] * sd_t)),
                                        "width_scaled": float(np.mean(2 * cfn["q"] * sd_t)),
                                        "spearman": float(spearmanr(sd_t, err).correlation),
                                        "rmse": float(np.sqrt(np.mean(err ** 2)))})
        print(f"ensemble seed {seed} done", flush=True)


# ------------------------------------------------------------------ generative
def _gen_setup():
    import generate_compare as G
    from models.surrogates import GBDTSurrogate
    ds = G.load_dataset()
    geom, fcol, tcol = ds["geom_cols"], ds["freq_col"], ds["target_col"]
    fwd = GBDTSurrogate(random_state=C.SEED).fit(ds["X"], ds["y"])    # fixed scorer for every seed
    lb = np.array([b[0] for b in ds["opt_bounds"]])
    ub = np.array([b[1] for b in ds["opt_bounds"]])
    return G, ds, geom, fcol, tcol, fwd, lb, ub


def _gen_row(G, name, seed, samp, ds, fwd, lb, ub, secs, **extra):
    m = G.metrics(samp, ds, fwd, lb, ub)
    return {"seed": seed, "method": name, "median": m["median"], "within3": m["within3"],
            "below_mask": m["below_mask"], "diversity": m["diversity"], "secs": secs, **extra}


def phase_generative():
    from generate.cgmm import ConditionalGMM
    from generate.cvae import CVAEInverseDesigner
    from generate.diffusion import DiffusionInverseDesigner
    G, ds, geom, fcol, tcol, fwd, lb, ub = _gen_setup()
    Xg = ds["df"][geom].to_numpy(float)
    Cc = ds["df"][[fcol, tcol]].to_numpy(float)
    Z = ds["df"][geom + [fcol, tcol]].to_numpy(float)
    target = np.array([ds["target_freq"], G.TARGET_S11])
    for seed in SEEDS5:
        if done("ms_generative.csv", seed=seed, method="Diffusion"):
            continue
        rng = np.random.default_rng(seed)
        t = time.perf_counter()
        samp = lb + rng.random((G.N_GEN, len(lb))) * (ub - lb)                  # no-model baseline
        append("ms_generative.csv", _gen_row(G, "Random", seed, samp, ds, fwd, lb, ub, time.perf_counter() - t))
        t = time.perf_counter()
        gmm = ConditionalGMM(n_components=40, seed=seed).fit(Z)
        samp = gmm.sample([len(geom), len(geom) + 1], target, n=G.N_GEN, seed=seed)[0]
        append("ms_generative.csv", _gen_row(G, "GMM", seed, samp, ds, fwd, lb, ub, time.perf_counter() - t))
        t = time.perf_counter()
        cvae = CVAEInverseDesigner(epochs=120, batch=1024, beta=0.5, seed=seed).fit(Xg, Cc)
        samp = cvae.sample(target, n=G.N_GEN, seed=seed)
        append("ms_generative.csv", _gen_row(G, "VAE", seed, samp, ds, fwd, lb, ub, time.perf_counter() - t))
        t = time.perf_counter()
        diff = DiffusionInverseDesigner(T=200, epochs=150, batch=1024, seed=seed).fit(Xg, Cc)
        samp = diff.sample(target, n=G.N_GEN, seed=seed)
        append("ms_generative.csv", _gen_row(G, "Diffusion", seed, samp, ds, fwd, lb, ub, time.perf_counter() - t))
        print(f"generative seed {seed} done", flush=True)


def phase_guidance():
    import generate_guidance as GG
    from generate.diffusion import DiffusionInverseDesigner
    G, ds, geom, fcol, tcol, fwd, lb, ub = _gen_setup()
    Xg = ds["df"][geom].to_numpy(float)
    Cc = ds["df"][[fcol, tcol]].to_numpy(float)
    target = np.array([ds["target_freq"], G.TARGET_S11])
    for seed in SEEDS5:
        if done("ms_guidance.csv", seed=seed, w=GG.SCALES[-1]):
            continue
        t = time.perf_counter()
        diff = DiffusionInverseDesigner(T=200, epochs=150, batch=1024, p_uncond=0.15, seed=seed).fit(Xg, Cc)
        secs = time.perf_counter() - t
        for w in GG.SCALES:
            samp = diff.sample(target, n=G.N_GEN, guidance_scale=w, seed=seed)
            append("ms_guidance.csv", _gen_row(G, "Diffusion+CFG", seed, samp, ds, fwd, lb, ub, secs, w=w))
        print(f"guidance seed {seed} done", flush=True)


def phase_curated():
    import generate_curated as GC
    from generate.cgmm import ConditionalGMM
    from generate.diffusion import DiffusionInverseDesigner
    from generate_multi import tl_s11, metrics, synth_joint
    tf, ts = GC.TARGET_F, GC.TARGET_S11
    for seed in SEEDS5:
        if done("ms_curated.csv", seed=seed, data="curated", method="Diffusion+CFG"):
            continue
        for Z, tag in ((synth_joint(seed=seed), "broad"), (GC.curated_joint(seed=seed), "curated")):
            lb, ub = Z[:, :5].min(0), Z[:, :5].max(0)
            rng = np.random.default_rng(seed)
            samples = {"Random": lb + rng.random((GC.N_GEN, 5)) * (ub - lb)}
            gmm = ConditionalGMM(n_components=55, seed=seed).fit(Z)
            samples["GMM"] = gmm.sample([5, 6], [tf, ts], n=GC.N_GEN, seed=seed)[0]
            diff = DiffusionInverseDesigner(T=200, epochs=180, batch=1024, p_uncond=0.15, seed=seed)
            diff.fit(Z[:, :5], Z[:, 5:7])
            samples["Diffusion+CFG"] = diff.sample([tf, ts], n=GC.N_GEN, guidance_scale=4.0, seed=seed)
            for name, s in samples.items():
                s = np.clip(s, lb, ub)
                ach = tl_s11(s[:, 0], s[:, 1], s[:, 2], s[:, 3], s[:, 4], tf)
                m = metrics(s, ach, lb, ub)
                append("ms_curated.csv", {"seed": seed, "data": tag, "method": name, "median": m["median"],
                                           "below_mask": m["below_mask"], "diversity": m["diversity"]})
        print(f"curated seed {seed} done", flush=True)


PHASES = {"surrogates": phase_surrogates, "uq": phase_uq, "ensemble": phase_ensemble,
          "generative": phase_generative, "guidance": phase_guidance, "curated": phase_curated}

if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(f"=== {p} ===", flush=True)
        PHASES[p]()
