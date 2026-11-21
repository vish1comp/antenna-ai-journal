"""Strengthening experiments for the improved journal manuscript.

Separate from exp_multiseed.py: every output goes to results/improved/, so the
results behind the v20 manuscript are never touched.

  python exp_extend.py uqbase    # UQ baselines: GBDT ensemble (ours) vs MLP deep ensemble,
                                 #   MC dropout, conformalized quantile GBDT (CQR), GP;
                                 #   raw and split-conformal intervals; 3 datasets x 5 seeds
  python exp_extend.py invbase   # inverse-design baselines on dataset C (exact physics):
                                 #   random, multi-start PSO on a surrogate, tandem network,
                                 #   GMM, diffusion+CFG; broad and curated data, 5 seeds
  python exp_extend.py bsize     # dataset B learning curve: is B data- or model-limited?
  python exp_extend.py cst       # trusted, diverse diffusion designs on dataset A for
                                 #   full-wave (CST/HFSS) validation by the authors
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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

import config as C

warnings.filterwarnings("ignore")
OUT = os.path.join(C.RESULTS_DIR, "improved")
os.makedirs(OUT, exist_ok=True)
SEEDS5 = [42, 7, 123, 2024, 11]
Z90 = 1.6448536269514722


def append(name, row):
    path = os.path.join(OUT, name)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def done(name, **keys):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        return False
    df = pd.read_csv(path)
    m = np.ones(len(df), bool)
    for k, v in keys.items():
        m &= df[k].astype(str).to_numpy() == str(v)
    return bool(m.any())


def conformal_q(scores, alpha=0.1):
    n = len(scores)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level, method="higher"))


# ------------------------------------------------------------------ torch helpers
def _torch():
    import torch
    torch.set_num_threads(4)
    return torch


class MCDropoutMLP:
    """2 x 128 ReLU MLP with dropout kept on at test time (Gal & Ghahramani)."""

    def __init__(self, p=0.1, epochs=150, batch=512, lr=1e-3, passes=50, seed=0):
        self.p, self.epochs, self.batch, self.lr, self.passes, self.seed = p, epochs, batch, lr, passes, seed

    def fit(self, X, y):
        torch = _torch()
        nn = torch.nn
        torch.manual_seed(self.seed)
        self.xm, self.xs = X.mean(0), X.std(0) + 1e-9
        self.ym, self.ys = y.mean(), y.std() + 1e-9
        Xs = torch.tensor((X - self.xm) / self.xs, dtype=torch.float32)
        ys = torch.tensor((y - self.ym) / self.ys, dtype=torch.float32)[:, None]
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(Xs))
        nv = max(1, int(0.15 * len(idx)))
        va, tr = idx[:nv], idx[nv:]
        d = X.shape[1]
        self.net = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Dropout(self.p),
                                 nn.Linear(128, 128), nn.ReLU(), nn.Dropout(self.p), nn.Linear(128, 1))
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        best, best_state, bad = np.inf, None, 0
        for _ in range(self.epochs):
            self.net.train()
            perm = tr[rng.permutation(len(tr))]
            for i in range(0, len(perm), self.batch):
                b = perm[i:i + self.batch]
                opt.zero_grad()
                loss = ((self.net(Xs[b]) - ys[b]) ** 2).mean()
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                v = float(((self.net(Xs[va]) - ys[va]) ** 2).mean())
            if v < best - 1e-5:
                best, bad = v, 0
                best_state = {k: t.clone() for k, t in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= 20:
                    break
        self.net.load_state_dict(best_state)
        return self

    def predict_mean_std(self, X):
        torch = _torch()
        Xs = torch.tensor((X - self.xm) / self.xs, dtype=torch.float32)
        self.net.train()                                   # dropout on
        torch.manual_seed(self.seed + 1)
        with torch.no_grad():
            S = np.stack([self.net(Xs).numpy().ravel() for _ in range(self.passes)])
        S = S * self.ys + self.ym
        return S.mean(0), S.std(0)


# ------------------------------------------------------------------ UQ baselines
def _uq_datasets():
    import benchmark_datasets as B
    from data.get_dataset import load_dataset
    ds = load_dataset()
    yield "A", ds["X"], ds["y"]                            # full dataset A, as in the UQ section
    for tag, loader in (("B", B.load_hf), ("C", B.gen_synthetic)):
        X, y, _ = loader()
        X, y = B._subsample(X, y, B.N_MAX)
        yield tag, X, y


def _uq_row(tag, seed, method, y_te, mu_t, lo_raw, hi_raw, lo_c, hi_c, score, fit_s):
    err = np.abs(y_te - mu_t)
    return {"dataset": tag, "seed": seed, "method": method,
            "cov_raw": float(np.mean((y_te >= lo_raw) & (y_te <= hi_raw))),
            "width_raw": float(np.mean(hi_raw - lo_raw)),
            "cov_conf": float(np.mean((y_te >= lo_c) & (y_te <= hi_c))),
            "width_conf": float(np.mean(hi_c - lo_c)),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "spearman": float(spearmanr(score, err).correlation),
            "fit_s": fit_s}


def phase_uqbase():
    from models.uncertainty import EnsembleSurrogate
    from models.surrogates import MLPSurrogate, GPSurrogate
    for tag, X, y in _uq_datasets():
        for seed in SEEDS5:
            if done("uq_baselines.csv", dataset=tag, seed=seed, method="GP"):
                continue
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=C.TEST_SIZE, random_state=seed)
            Xf, Xc, yf, yc = train_test_split(X_tr, y_tr, test_size=0.25, random_state=seed)

            def sigma_method(name, fit_fn):
                t = time.perf_counter()
                predict = fit_fn()
                fit_s = time.perf_counter() - t
                mu_c, sd_c = predict(Xc)
                mu_t, sd_t = predict(X_te)
                q = conformal_q(np.abs(yc - mu_c) / np.maximum(sd_c, 1e-9))
                append("uq_baselines.csv", _uq_row(tag, seed, name, y_te, mu_t,
                                                   mu_t - Z90 * sd_t, mu_t + Z90 * sd_t,
                                                   mu_t - q * sd_t, mu_t + q * sd_t, sd_t, fit_s))

            def gbdt_ens():
                e = EnsembleSurrogate(n_models=C.UQ["n_models"], seed=seed).fit(Xf, yf)
                return e.predict_mean_std
            sigma_method("GBDT ensemble (ours)", gbdt_ens)

            def mlp_ens():
                ms = [MLPSurrogate(random_state=seed * 10 + i).fit(Xf, yf) for i in range(5)]

                def pr(Z):
                    S = np.stack([m.predict(Z) for m in ms])
                    return S.mean(0), S.std(0)
                return pr
            sigma_method("MLP deep ensemble", mlp_ens)

            def mcd():
                return MCDropoutMLP(seed=seed).fit(Xf, yf).predict_mean_std
            sigma_method("MC dropout", mcd)

            # conformalized quantile regression with GBDT quantile models
            t = time.perf_counter()
            kw = dict(learning_rate=0.08, max_iter=500, l2_regularization=1.0, early_stopping=True,
                      random_state=seed)
            qlo = HistGradientBoostingRegressor(loss="quantile", quantile=0.05, **kw).fit(Xf, yf)
            qhi = HistGradientBoostingRegressor(loss="quantile", quantile=0.95, **kw).fit(Xf, yf)
            med = HistGradientBoostingRegressor(loss="quantile", quantile=0.5, **kw).fit(Xf, yf)
            fit_s = time.perf_counter() - t
            lc, hc = qlo.predict(Xc), qhi.predict(Xc)
            q = conformal_q(np.maximum(lc - yc, yc - hc))
            lt, ht = qlo.predict(X_te), qhi.predict(X_te)
            append("uq_baselines.csv", _uq_row(tag, seed, "CQR (quantile GBDT)", y_te, med.predict(X_te),
                                               lt, ht, lt - q, ht + q, ht - lt, fit_s))

            def gp():
                idx = np.random.default_rng(seed).choice(len(Xf), min(C.GP_MAX_TRAIN, len(Xf)), replace=False)
                g = GPSurrogate(random_state=seed).fit(Xf[idx], yf[idx])
                return lambda Z: (g.predict(Z), g.predict_std(Z))
            sigma_method("GP", gp)
            print(f"uqbase {tag} seed {seed} done", flush=True)


# ------------------------------------------------------------------ inverse-design baselines
class Tandem:
    """Tandem network (Liu et al. 2018): an inverse net c -> x trained through a frozen
    forward net, so it returns one design per condition (no diversity by construction)."""

    def __init__(self, lb, ub, seed=0, epochs=60):
        self.lb, self.ub, self.seed, self.epochs = lb, ub, seed, epochs

    def fit(self, Z):
        torch = _torch()
        nn = torch.nn
        torch.manual_seed(self.seed)
        Xg, f, s = Z[:, :5], Z[:, 5:6], Z[:, 6:7]
        self.fm, self.fsd = f.mean(), f.std() + 1e-9
        self.sm, self.ssd = s.mean(), s.std() + 1e-9
        xn = (Xg - self.lb) / (self.ub - self.lb)
        T = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
        xin = T(np.hstack([xn, (f - self.fm) / self.fsd]))
        tgt = T((s - self.sm) / self.ssd)
        fwd = nn.Sequential(nn.Linear(6, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
        inv = nn.Sequential(nn.Linear(2, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 5),
                            nn.Sigmoid())
        rng = np.random.default_rng(self.seed)
        opt = torch.optim.Adam(fwd.parameters(), lr=1e-3)
        for _ in range(self.epochs):                       # 1) forward model
            p = rng.permutation(len(xin))
            for i in range(0, len(p), 512):
                b = p[i:i + 512]
                opt.zero_grad()
                ((fwd(xin[b]) - tgt[b]) ** 2).mean().backward()
                opt.step()
        for q in fwd.parameters():
            q.requires_grad_(False)
        cond = torch.cat([xin[:, 5:6], tgt], 1)
        opt = torch.optim.Adam(inv.parameters(), lr=1e-3)
        for _ in range(self.epochs):                       # 2) inverse through frozen forward
            p = rng.permutation(len(cond))
            for i in range(0, len(p), 512):
                b = p[i:i + 512]
                opt.zero_grad()
                xh = inv(cond[b])
                ((fwd(torch.cat([xh, cond[b, :1]], 1)) - cond[b, 1:]) ** 2).mean().backward()
                opt.step()
        self.inv = inv
        return self

    def design(self, f, s):
        torch = _torch()
        c = torch.tensor([[(f - self.fm) / self.fsd, (s - self.sm) / self.ssd]], dtype=torch.float32)
        with torch.no_grad():
            xn = self.inv(c).numpy()[0]
        return self.lb + xn * (self.ub - self.lb)


def phase_invbase():
    import generate_curated as GC
    from generate.cgmm import ConditionalGMM
    from generate.diffusion import DiffusionInverseDesigner
    from generate_multi import tl_s11, synth_joint, MASK
    from models.surrogates import MLPSurrogate
    from optimize.pso import particle_swarm
    tf, ts, N = GC.TARGET_F, GC.TARGET_S11, GC.N_GEN
    R = 40                                                 # PSO restarts = designs returned
    for seed in SEEDS5:
        for Z, data in ((synth_joint(seed=seed), "broad"), (GC.curated_joint(seed=seed), "curated")):
            if done("inv_baselines.csv", seed=seed, data=data, method="Tandem"):
                continue
            lb, ub = Z[:, :5].min(0), Z[:, :5].max(0)
            runs = {}
            rng = np.random.default_rng(seed)
            t = time.perf_counter()
            runs["Random"] = (lb + rng.random((N, 5)) * (ub - lb), time.perf_counter() - t, 0)
            t = time.perf_counter()
            fwd = MLPSurrogate(random_state=seed).fit(Z[:, :6], Z[:, 6])
            fit = time.perf_counter() - t
            t = time.perf_counter()
            P = []
            for r in range(R):
                g = particle_swarm(lambda X: np.abs(fwd.predict(np.hstack([X, np.full((len(X), 1), tf)])) - ts),
                                   list(zip(lb, ub)), minimize=True, seed=seed * 100 + r, **C.UQ_PSO)
                P.append(g["x"])
            evals = R * C.UQ_PSO["n_particles"] * C.UQ_PSO["n_iters"]
            runs["Multi-start PSO"] = (np.array(P), fit + time.perf_counter() - t, evals)
            t = time.perf_counter()
            tan = Tandem(lb, ub, seed=seed).fit(Z)
            runs["Tandem"] = (tan.design(tf, ts)[None, :], time.perf_counter() - t, 0)
            t = time.perf_counter()
            gmm = ConditionalGMM(n_components=55, seed=seed).fit(Z)
            runs["GMM"] = (gmm.sample([5, 6], [tf, ts], n=N, seed=seed)[0], time.perf_counter() - t, 0)
            t = time.perf_counter()
            diff = DiffusionInverseDesigner(T=200, epochs=180, batch=1024, p_uncond=0.15, seed=seed)
            diff.fit(Z[:, :5], Z[:, 5:7])
            runs["Diffusion+CFG"] = (diff.sample([tf, ts], n=N, guidance_scale=4.0, seed=seed),
                                     time.perf_counter() - t, 0)
            for name, (S, secs, ev) in runs.items():
                S = np.clip(S, lb, ub)
                ach = tl_s11(S[:, 0], S[:, 1], S[:, 2], S[:, 3], S[:, 4], tf)
                div = float(np.mean(S.std(0) / (ub - lb + 1e-9))) if len(S) > 1 else 0.0
                append("inv_baselines.csv", {"seed": seed, "data": data, "method": name, "n_designs": len(S),
                                             "median": float(np.median(ach)),
                                             "hit3": float(np.mean(np.abs(ach - ts) <= 3.0)),
                                             "below_mask": float(np.mean(ach <= MASK)),
                                             "diversity": div, "secs": secs,
                                             "secs_per_design": secs / len(S), "surrogate_evals": ev})
            print(f"invbase seed {seed} {data} done", flush=True)


# ------------------------------------------------------------------ dataset-B learning curve
def phase_bsize():
    import benchmark_datasets as B
    from evaluate.metrics import regression_report
    from models.surrogates import GBDTSurrogate, MLPSurrogate
    X, y, _ = B.load_hf()
    for seed in SEEDS5[:3]:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X))
        te, pool = idx[:40000], idx[40000:]
        for n in (1500, 6000, 24000, 96000, 300000):
            if done("b_learning_curve.csv", seed=seed, n_train=n, model="MLP"):
                continue
            tr = pool[:n]
            for name, mdl in (("GBDT", GBDTSurrogate(random_state=seed)), ("MLP", MLPSurrogate(random_state=seed))):
                t = time.perf_counter()
                mdl.fit(X[tr], y[tr])
                rep = regression_report(y[te], mdl.predict(X[te]))
                append("b_learning_curve.csv", {"seed": seed, "n_train": n, "model": name, "R2": rep["R2"],
                                                "NRMSE": rep["NRMSE"], "fit_s": time.perf_counter() - t})
            print(f"bsize seed {seed} n {n} done", flush=True)


# ------------------------------------------------------------------ CST validation list
def phase_cst():
    import generate_compare as G
    from data.get_dataset import load_dataset
    from generate.diffusion import DiffusionInverseDesigner
    from models.uncertainty import EnsembleSurrogate
    from sklearn.cluster import KMeans
    ds = load_dataset()
    geom, fcol, tcol = ds["geom_cols"], ds["freq_col"], ds["target_col"]
    lb = np.array([b[0] for b in ds["opt_bounds"]])
    ub = np.array([b[1] for b in ds["opt_bounds"]])
    seed = C.SEED
    X_tr, X_te, y_tr, y_te = train_test_split(ds["X"], ds["y"], test_size=C.TEST_SIZE, random_state=seed)
    Xf, Xc, yf, yc = train_test_split(X_tr, y_tr, test_size=0.25, random_state=seed)
    ens = EnsembleSurrogate(n_models=C.UQ["n_models"], seed=seed).fit(Xf, yf)
    mu_c, sd_c = ens.predict_mean_std(Xc)
    q = conformal_q(np.abs(yc - mu_c) / np.maximum(sd_c, 1e-9))
    Xg = ds["df"][geom].to_numpy(float)
    Cc = ds["df"][[fcol, tcol]].to_numpy(float)
    diff = DiffusionInverseDesigner(T=200, epochs=150, batch=1024, p_uncond=0.15, seed=seed).fit(Xg, Cc)
    S = diff.sample(np.array([ds["target_freq"], G.TARGET_S11]), n=2000, guidance_scale=3.0, seed=seed)
    S = np.round(np.clip(S, lb, ub), 1)                    # 0.1 mm: a buildable geometry
    S = np.unique(S, axis=0)
    mu, sd = ens.predict_mean_std(ds["assemble"](S))
    upper = mu + q * sd
    ok = upper <= -10.0                                    # trust gate
    T, mu, sd, upper = S[ok], mu[ok], sd[ok], upper[ok]
    # distance to the closest real (CST-simulated) geometry, in units of the range
    real = np.unique(ds["verify_X"], axis=0)
    dn = np.array([np.min(np.linalg.norm((real - t) / (ub - lb), axis=1)) for t in T])
    km = KMeans(n_clusters=min(5, len(T)), n_init=10, random_state=seed).fit((T - lb) / (ub - lb))
    pick = []
    for k in range(km.n_clusters):                         # best trusted design of each cluster
        members = np.where(km.labels_ == k)[0]
        pick.append(members[np.argmin(upper[members])])
    rows = []
    for j, i in enumerate(sorted(pick, key=lambda i: upper[i]), 1):
        r = {"design": f"D{j}"}
        r.update({c: float(v) for c, v in zip(geom, T[i])})
        r.update({"pred_S11_2p4_dB": round(float(mu[i]), 2), "sigma_dB": round(float(sd[i]), 2),
                  "interval_lo_dB": round(float(mu[i] - q * sd[i]), 2),
                  "interval_hi_dB": round(float(upper[i]), 2),
                  "dist_to_nearest_real": round(float(dn[i]), 3)})
        rows.append(r)
    # reference: the best real CST design, so the authors can check their own model first
    b = int(np.argmin(ds["verify_y"]))
    ref = {"design": "REF (dataset)"}
    ref.update({c: float(v) for c, v in zip(geom, ds["verify_X"][b])})
    rm, rs = ens.predict_mean_std(ds["assemble"](ds["verify_X"][b][None, :]))
    ref.update({"pred_S11_2p4_dB": round(float(rm[0]), 2), "sigma_dB": round(float(rs[0]), 2),
                "interval_lo_dB": round(float(rm[0] - q * rs[0]), 2),
                "interval_hi_dB": round(float(rm[0] + q * rs[0]), 2), "dist_to_nearest_real": 0.0,
                "dataset_S11_2p4_dB": round(float(ds["verify_y"][b]), 2)})
    rows.append(ref)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "cst_designs.csv"), index=False)
    # predicted sweeps with intervals, 1.8-2.8 GHz, for overlaying the CST result
    freqs = np.round(np.arange(1.8, 2.8001, 0.01), 3)
    sweep = {"freq_GHz": freqs}
    for r, g in zip(rows, [T[i] for i in sorted(pick, key=lambda i: upper[i])] + [ds["verify_X"][b]]):
        m, s = ens.predict_mean_std(np.vstack([ds["assemble_at"](g[None, :], f) for f in freqs]))
        sweep[f"{r['design']}_mu"] = np.round(m, 3)
        sweep[f"{r['design']}_lo"] = np.round(m - q * s, 3)
        sweep[f"{r['design']}_hi"] = np.round(m + q * s, 3)
    pd.DataFrame(sweep).to_csv(os.path.join(OUT, "cst_predicted_sweeps.csv"), index=False)
    print(f"cst: {ok.sum()} of {len(S)} unique buildable samples pass the gate; q = {q:.3f}")
    print(pd.DataFrame(rows).to_string(index=False))


PHASES = {"uqbase": phase_uqbase, "invbase": phase_invbase, "bsize": phase_bsize, "cst": phase_cst}

if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(f"=== {p} ===", flush=True)
        PHASES[p]()
