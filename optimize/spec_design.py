"""Inverse spec-matching design: shape the S11(f) response, not just one point.

Given a surrogate S(geometry, f) -> S11(dB), design the geometry whose predicted
response sits below a return-loss mask across a target band and resonates at the
band center. This is the real antenna-design objective (a bandwidth spec), a step
up from minimizing S11 at a single frequency.
"""
from __future__ import annotations

import numpy as np


def make_spec(center=2.4, band=(2.35, 2.45), n_band=11, mask_db=-10.0,
              band_weight=4.0, plot_range=(2.20, 2.60), n_plot=81):
    """Target: S11 <= mask_db across `band`, resonant at `center`."""
    return {
        "center": center,
        "band": band,
        "freqs": np.linspace(band[0], band[1], n_band),   # used by the loss
        "mask_db": mask_db,
        "band_weight": band_weight,
        "plot_freqs": np.linspace(plot_range[0], plot_range[1], n_plot),
        "plot_range": plot_range,
    }


def _response(surrogate, ds, P_geom, freqs):
    """Predicted S11 (n_particles, n_freq) for each geometry across `freqs`."""
    P_geom = np.atleast_2d(np.asarray(P_geom, float))
    cols = [surrogate.predict(ds["assemble_at"](P_geom, fv)) for fv in freqs]
    return np.vstack(cols).T  # (n_particles, n_freq)


def spec_objective(surrogate, ds, spec):
    """Return f(P_geom)->(n,) loss: deep at center + penalty for exceeding the mask."""
    freqs = spec["freqs"]
    mask = spec["mask_db"]
    w = spec["band_weight"]
    ci = int(np.argmin(np.abs(freqs - spec["center"])))

    def f(P_geom):
        R = _response(surrogate, ds, P_geom, freqs)          # (n, F)
        center = R[:, ci]                                    # want very negative
        violation = np.maximum(0.0, R - mask).mean(axis=1)   # dB above the mask
        return center + w * violation                        # minimize
    return f


def evaluate_response(surrogate, ds, spec, x_geom):
    """Predicted response curve + achieved metrics for one designed geometry."""
    freqs = spec["plot_freqs"]
    R = _response(surrogate, ds, np.atleast_2d(x_geom), freqs)[0]
    below = R <= spec["mask_db"]
    ci = int(np.argmin(np.abs(freqs - spec["center"])))
    lo = hi = ci
    if below[ci]:
        while lo - 1 >= 0 and below[lo - 1]:
            lo -= 1
        while hi + 1 < len(freqs) and below[hi + 1]:
            hi += 1
        bw = float(freqs[hi] - freqs[lo])
    else:
        bw = 0.0
    return {
        "freqs": freqs, "s11": R,
        "center_s11": float(R[ci]),
        "min_s11": float(R.min()),
        "bandwidth_ghz": bw,
    }


def nearest_true_curve(ds, x_geom):
    """True measured S11 sweep of the dataset geometry nearest x_geom (verification)."""
    df = ds["df"]
    G = df[ds["geom_cols"]].to_numpy(float)
    uniq, inv = np.unique(G, axis=0, return_inverse=True)
    j = int(np.argmin(np.linalg.norm(uniq - np.asarray(x_geom, float), axis=1)))
    rows = df[inv == j].sort_values(ds["freq_col"])
    return (rows[ds["freq_col"]].to_numpy(float),
            rows[ds["target_col"]].to_numpy(float),
            float(np.linalg.norm(uniq[j] - np.asarray(x_geom, float))))
