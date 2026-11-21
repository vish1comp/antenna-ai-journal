"""Conditional denoising diffusion (DDPM) with classifier-free guidance (CFG).

The current state of the art for generative inverse design (MetaDiffusion 2023).
A network learns to denoise geometry vectors; conditioned on a target
(frequency, S11), ancestral sampling from noise yields diverse geometries meeting
the spec.

Classifier-free guidance adds a tunable fidelity <-> diversity knob: during
training the condition is randomly dropped (p_uncond) so the model learns both a
conditional and an unconditional denoiser; at sampling the two are combined,
    eps = eps_uncond + w * (eps_cond - eps_uncond),
where the guidance scale w > 1 sharpens adherence to the spec at the cost of
diversity (w = 1 recovers the plain conditional model).

CPU-only PyTorch; a small MLP epsilon-network suffices for a 7-D geometry vector.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import torch
import torch.nn as nn


def _time_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-np.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
    ang = t[:, None].float() * freqs[None, :]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)


class _EpsNet(nn.Module):
    """Predicts the noise from [x_t, masked condition, mask flag, time]."""

    def __init__(self, d_x, d_c, d_t=16, h=128):
        super().__init__()
        self.d_t = d_t
        self.net = nn.Sequential(
            nn.Linear(d_x + d_c + 1 + d_t, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, d_x))

    def forward(self, x, t, c, mask):
        # mask: (n,1) in {0,1}; when 0 the condition is hidden (unconditional).
        inp = torch.cat([x, c * mask, mask, _time_embedding(t, self.d_t)], dim=1)
        return self.net(inp)


class DiffusionInverseDesigner:
    def __init__(self, T=200, epochs=150, batch=1024, lr=1e-3, d_t=16, hidden=128,
                 p_uncond=0.15, seed=0):
        self.T, self.epochs, self.batch = T, epochs, batch
        self.lr, self.d_t, self.hidden = lr, d_t, hidden
        self.p_uncond, self.seed = p_uncond, seed

    def _std(self, A, m, s):
        return (A - m) / s

    def fit(self, X_geom, C_cond, verbose=False):
        X = np.asarray(X_geom, float)
        Cc = np.asarray(C_cond, float)
        self.xm_, self.xs_ = X.mean(0), X.std(0) + 1e-9
        self.cm_, self.cs_ = Cc.mean(0), Cc.std(0) + 1e-9
        Xs = torch.tensor(self._std(X, self.xm_, self.xs_), dtype=torch.float32)
        Cs = torch.tensor(self._std(Cc, self.cm_, self.cs_), dtype=torch.float32)

        betas = torch.linspace(1e-4, 0.02, self.T)
        self.betas, self.alphas = betas, 1.0 - betas
        self.abar = torch.cumprod(self.alphas, dim=0)

        torch.manual_seed(self.seed)
        self.model = _EpsNet(X.shape[1], Cc.shape[1], self.d_t, self.hidden)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        n = len(Xs)
        g = torch.Generator().manual_seed(self.seed)
        for ep in range(self.epochs):
            perm = torch.randperm(n, generator=g)
            tot = 0.0
            for i in range(0, n, self.batch):
                idx = perm[i:i + self.batch]
                x0, c = Xs[idx], Cs[idx]
                t = torch.randint(0, self.T, (len(idx),), generator=g)
                ab = self.abar[t][:, None]
                noise = torch.randn(x0.shape, generator=g)
                xt = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise
                # drop the condition with prob p_uncond (learn the uncond. denoiser)
                mask = (torch.rand(len(idx), 1, generator=g) >= self.p_uncond).float()
                pred = self.model(xt, t, c, mask)
                loss = ((pred - noise) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss) * len(idx)
            if verbose and (ep + 1) % 30 == 0:
                print(f"    epoch {ep+1:3d}  denoise MSE={tot/n:.4f}")
        return self

    @torch.no_grad()
    def sample(self, cond_val, n=400, guidance_scale=1.0, seed=0):
        c = self._std(np.asarray(cond_val, float)[None, :], self.cm_, self.cs_)
        Cs = torch.tensor(np.repeat(c, n, axis=0), dtype=torch.float32)
        one = torch.ones(n, 1)
        zero = torch.zeros(n, 1)
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(n, len(self.xm_), generator=g)
        for t in reversed(range(self.T)):
            tt = torch.full((n,), t, dtype=torch.long)
            eps_c = self.model(x, tt, Cs, one)
            if guidance_scale == 1.0:
                eps = eps_c
            else:
                eps_u = self.model(x, tt, Cs, zero)
                eps = eps_u + guidance_scale * (eps_c - eps_u)
            a, ab, b = self.alphas[t], self.abar[t], self.betas[t]
            mean = (x - (b / torch.sqrt(1 - ab)) * eps) / torch.sqrt(a)
            x = mean + (torch.sqrt(b) * torch.randn(x.shape, generator=g) if t > 0 else 0.0)
        return x.numpy() * self.xs_ + self.xm_
