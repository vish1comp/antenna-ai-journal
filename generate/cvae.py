"""Conditional Variational Autoencoder (cVAE) for generative inverse design.

A deep generative counterpart to the conditional GMM in `cgmm.py`. The encoder
q(z | geometry, condition) and decoder p(geometry | z, condition) are trained on
the joint data; to inverse-design, draw z ~ N(0, I) and decode with the target
condition, yielding diverse geometries that meet the spec. This is the class of
model (Naseri & Hum 2021 use a VAE) the survey argues antenna engineering should
adopt from photonics.

CPU-only PyTorch. The KMP flag sidesteps the Anaconda/MKL + PyTorch OpenMP clash.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import torch
import torch.nn as nn


class _CVAE(nn.Module):
    def __init__(self, d_x, d_c, d_z=4, h=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_x + d_c, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU())
        self.mu = nn.Linear(h, d_z)
        self.lv = nn.Linear(h, d_z)
        self.dec = nn.Sequential(nn.Linear(d_z + d_c, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, d_x))
        self.d_z = d_z

    def encode(self, x, c):
        h = self.enc(torch.cat([x, c], dim=1))
        return self.mu(h), self.lv(h)

    def decode(self, z, c):
        return self.dec(torch.cat([z, c], dim=1))

    def forward(self, x, c):
        mu, lv = self.encode(x, c)
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(lv)
        return self.decode(z, c), mu, lv


class CVAEInverseDesigner:
    """sklearn-flavored wrapper: fit(X_geom, C_cond) then sample(cond_val, n)."""

    def __init__(self, d_z=4, hidden=64, epochs=120, batch=1024, beta=0.5,
                 lr=1e-3, seed=0):
        self.d_z, self.hidden = d_z, hidden
        self.epochs, self.batch, self.beta, self.lr, self.seed = epochs, batch, beta, lr, seed

    def _std(self, A, m, s):
        return (A - m) / s

    def fit(self, X_geom, C_cond, verbose=False):
        X = np.asarray(X_geom, float)
        C = np.asarray(C_cond, float)
        self.xm_, self.xs_ = X.mean(0), X.std(0) + 1e-9
        self.cm_, self.cs_ = C.mean(0), C.std(0) + 1e-9
        Xs = torch.tensor(self._std(X, self.xm_, self.xs_), dtype=torch.float32)
        Cs = torch.tensor(self._std(C, self.cm_, self.cs_), dtype=torch.float32)

        torch.manual_seed(self.seed)
        self.model = _CVAE(X.shape[1], C.shape[1], self.d_z, self.hidden)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        n = len(Xs)
        g = torch.Generator().manual_seed(self.seed)
        for ep in range(self.epochs):
            perm = torch.randperm(n, generator=g)
            tot = 0.0
            for i in range(0, n, self.batch):
                idx = perm[i:i + self.batch]
                xb, cb = Xs[idx], Cs[idx]
                recon, mu, lv = self.model(xb, cb)
                rec = ((recon - xb) ** 2).mean()
                kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
                loss = rec + self.beta * kl
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss) * len(idx)
            if verbose and (ep + 1) % 30 == 0:
                print(f"    epoch {ep+1:3d}  ELBO loss={tot/n:.4f}")
        return self

    @torch.no_grad()
    def sample(self, cond_val, n=400, seed=0):
        c = self._std(np.asarray(cond_val, float)[None, :], self.cm_, self.cs_)
        Cs = torch.tensor(np.repeat(c, n, axis=0), dtype=torch.float32)
        g = torch.Generator().manual_seed(seed)
        z = torch.randn(n, self.d_z, generator=g)
        x = self.model.decode(z, Cs).numpy()
        return x * self.xs_ + self.xm_        # un-standardize geometry
