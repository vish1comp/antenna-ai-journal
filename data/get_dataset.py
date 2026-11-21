"""Load the public patch-antenna dataset and shape it for the demo.

Dataset: "Simulated Patch Antenna for 2.4 GHz Applications" (A. Mersani),
Zenodo record 17527112, CC-BY-4.0. ~55k rows of microstrip-patch geometry +
frequency -> S11 (dB), generated in CST Microwave Studio.

We frame two tasks from one dataset:
  * SURROGATE:     predict S11(dB) from 7 geometry params + frequency.
  * OPTIMIZATION:  design the geometry that minimizes S11 at 2.4 GHz
                   (frequency pinned to the application band).

The CSV is cached under data/raw/ after the first download.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

DATA_URL = "https://zenodo.org/api/records/17527112/files/DataSet.csv/content"
DATA_URL_ALT = "https://zenodo.org/records/17527112/files/DataSet.csv?download=1"
RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
RAW_CSV = os.path.join(RAW_DIR, "DataSet.csv")

TARGET = "S11 (dB)"
FREQ_COL = "Frequency (GHz)"
TARGET_FREQ = 2.4  # application band (GHz)


def _download(path):
    os.makedirs(RAW_DIR, exist_ok=True)
    import requests
    last = None
    for url in (DATA_URL, DATA_URL_ALT):
        try:
            print(f"  downloading {url}")
            r = requests.get(url, timeout=90)
            r.raise_for_status()
            if b";" not in r.content[:2000] and b"," not in r.content[:2000]:
                raise ValueError("downloaded file does not look like a CSV")
            with open(path, "wb") as f:
                f.write(r.content)
            return
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"    failed: {e}")
    raise RuntimeError(f"could not download dataset: {last}")


def load_dataset():
    if not os.path.exists(RAW_CSV):
        _download(RAW_CSV)
    df = pd.read_csv(RAW_CSV, sep=";")
    df.columns = [c.strip() for c in df.columns]
    df = df.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)

    # keep only columns that actually vary; drop constants (feed width, FR4
    # thickness, copper cladding) which carry no information for ML.
    varying = [c for c in df.columns if df[c].nunique() > 1]
    features = [c for c in varying if c != TARGET]
    geom_features = [c for c in features if c != FREQ_COL]

    X = df[features].to_numpy(float)
    y = df[TARGET].to_numpy(float)
    bounds = [(float(df[c].min()), float(df[c].max())) for c in features]

    # --- optimization framing: fix frequency, search geometry only ---
    freqs = df[FREQ_COL].to_numpy(float)
    nearest_f = float(freqs[np.argmin(np.abs(freqs - TARGET_FREQ))])
    opt_bounds = [(float(df[c].min()), float(df[c].max())) for c in geom_features]
    fi = features.index(FREQ_COL)
    geom_idx = [features.index(c) for c in geom_features]

    def assemble(P_geom):
        """Map geometry particles (n, n_geom) -> full feature rows with freq pinned."""
        P_geom = np.atleast_2d(np.asarray(P_geom, float))
        full = np.empty((P_geom.shape[0], len(features)), float)
        full[:, geom_idx] = P_geom
        full[:, fi] = nearest_f
        return full

    def assemble_at(P_geom, freq_value):
        """Same as assemble but at an arbitrary frequency (for response sweeps)."""
        P_geom = np.atleast_2d(np.asarray(P_geom, float))
        full = np.empty((P_geom.shape[0], len(features)), float)
        full[:, geom_idx] = P_geom
        full[:, fi] = float(freq_value)
        return full

    # verification pool: real designs evaluated at (approximately) 2.4 GHz
    at_band = df[np.isclose(df[FREQ_COL], nearest_f)]
    verify_X = at_band[geom_features].to_numpy(float)
    verify_y = at_band[TARGET].to_numpy(float)

    return {
        "name": "Simulated Patch Antenna @ 2.4 GHz (Zenodo 17527112)",
        "source": "Zenodo 10.5281/zenodo.17527112, CC-BY-4.0",
        "X": X, "y": y,
        "feature_names": features,
        "target_name": TARGET,
        "direction": "min",            # minimize S11 (deeper resonance is better)
        "bounds": bounds,
        # optimization helpers
        "opt_feature_names": geom_features,
        "opt_bounds": opt_bounds,
        "assemble": assemble,
        "assemble_at": assemble_at,
        "verify_X": verify_X,
        "verify_y": verify_y,
        "target_freq": nearest_f,
        # frequency-sweep / spec-matching helpers
        "df": df,
        "geom_cols": geom_features,
        "freq_col": FREQ_COL,
        "target_col": TARGET,
        "geom_idx": geom_idx,
        "freq_index": fi,
    }


if __name__ == "__main__":
    ds = load_dataset()
    print(ds["name"])
    print("rows:", ds["X"].shape[0], "features:", ds["X"].shape[1])
    print("features:", ds["feature_names"])
    print("geometry (optimized):", ds["opt_feature_names"])
    print("verify pool @", round(ds["target_freq"], 4), "GHz:", ds["verify_X"].shape[0], "designs")
    print("S11 range:", round(ds["y"].min(), 2), "to", round(ds["y"].max(), 2), "dB")
