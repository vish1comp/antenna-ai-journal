"""Conditional generative inverse design via a joint Gaussian Mixture Model.

Fit a full-covariance GMM over the joint vector [geometry, frequency, S11]. To
inverse-design, condition the mixture on a target (frequency, S11) and sample
geometries from the resulting conditional mixture. Because the mixture is
multimodal, this yields a *diverse set* of candidate geometries that meet the
spec - the genuine one-to-many nature of inverse design - in contrast to the
single point an optimizer returns.

Conditioning a Gaussian mixture is closed-form: each component's Gaussian is
conditioned on the fixed dimensions, and the mixture weights are re-weighted by
how well each component explains the conditioning values.

VAE / GAN / diffusion models (Naseri & Hum 2021; MetaDiffusion 2023) are the
deep-learning extensions of this density-based idea to free-form, high-dimensional
geometry; this GMM is the transparent, dependency-light baseline.
"""
from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture


class ConditionalGMM:
    def __init__(self, n_components=30, reg_covar=1e-3, seed=0):
        self.n_components = n_components
        self.reg_covar = reg_covar
        self.seed = seed

    def fit(self, Z):
        Z = np.asarray(Z, float)
        self.mu_ = Z.mean(axis=0)
        self.sd_ = Z.std(axis=0) + 1e-9
        Zs = (Z - self.mu_) / self.sd_
        self.gmm = GaussianMixture(
            n_components=self.n_components, covariance_type="full",
            reg_covar=self.reg_covar, max_iter=300, random_state=self.seed).fit(Zs)
        self.d = Z.shape[1]
        return self

    def _condition(self, cond_idx, cond_val_s):
        """Per-component conditional mean/cov over free dims + re-weighted weights
        (all in standardized space). cond_val_s is already standardized."""
        free_idx = [i for i in range(self.d) if i not in cond_idx]
        means, covs, w = self.gmm.means_, self.gmm.covariances_, self.gmm.weights_
        c = np.asarray(cond_val_s, float)
        cm, cc, logw = [], [], []
        for k in range(self.gmm.n_components):
            mu, S = means[k], covs[k]
            mu_f, mu_c = mu[free_idx], mu[cond_idx]
            S_ff = S[np.ix_(free_idx, free_idx)]
            S_fc = S[np.ix_(free_idx, cond_idx)]
            S_cc = S[np.ix_(cond_idx, cond_idx)]
            diff = c - mu_c
            sol = np.linalg.solve(S_cc, diff)          # S_cc^{-1} (c - mu_c)
            gain = S_fc @ np.linalg.solve(S_cc, S_fc.T)  # S_fc S_cc^{-1} S_cf
            cm.append(mu_f + S_fc @ sol)
            S_cond = S_ff - gain
            cc.append(0.5 * (S_cond + S_cond.T) + 1e-6 * np.eye(len(free_idx)))
            sign, logdet = np.linalg.slogdet(2 * np.pi * S_cc)
            logw.append(np.log(w[k] + 1e-300) - 0.5 * (diff @ sol + logdet))
        logw = np.asarray(logw)
        logw -= logw.max()
        wn = np.exp(logw)
        wn /= wn.sum()
        return free_idx, np.asarray(cm), np.asarray(cc), wn

    def sample(self, cond_idx, cond_val, n=300, seed=0):
        """Sample n geometries conditioned on cond_val (raw units)."""
        cond_val_s = (np.asarray(cond_val, float) - self.mu_[cond_idx]) / self.sd_[cond_idx]
        free_idx, cm, cc, wn = self._condition(cond_idx, cond_val_s)
        rng = np.random.default_rng(seed)
        comp = rng.choice(len(wn), size=n, p=wn)
        samp_s = np.array([rng.multivariate_normal(cm[k], cc[k]) for k in comp])
        samp = samp_s * self.sd_[free_idx] + self.mu_[free_idx]   # un-standardize
        return samp, free_idx, wn
