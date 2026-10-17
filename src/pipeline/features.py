"""Feature engineering (Phase 2): per-window MCSA + time-domain features.

One row = one 256 ms window of BOTH phase currents. Features:
- time domain: rms, std, peak, crest, skewness, kurtosis (mean dropped: DC offset)
- spectral (index = Welch PSD): sideband ratios at LINE_FREQ +/- k*fault_hz and
  fault-energy share, for BPFO/BPFI/BSF/FTF per current channel.

Output: data/processed/features_window.csv
"""

import numpy as np
import pandas as pd

from utils.loaders import load_recording
from utils.features import time_domain_features, sliding_windows
from utils.spectral import (
    RATE_CUR,
    psd,
    fault_frequencies,
    sideband_ratio,
    fault_energy_share,
)
from pipeline.config import (
    PROCESSED_DIR,
    CURRENT_CHANNELS,
    FAULT_TYPES,
    SIDEBAND_HARMONICS,
    NPERSEG_FEAT,
    WINDOW,
    STEP,
)


def window_features(signal, speed_rpm):
    """Feature dict for one window of one current channel."""
    td = time_domain_features(signal)
    feats = {"rms": td["rms"], "std": td["std"], "peak": td["peak"],
             "crest": td["crest_factor"], "skew": td["skewness"], "kurt": td["kurtosis"]}

    f, p = psd(signal, RATE_CUR, nperseg=NPERSEG_FEAT)
    ff = fault_frequencies(speed_rpm / 60.0)
    for fault in FAULT_TYPES:
        for k in SIDEBAND_HARMONICS:
            lo, hi = sideband_ratio(f, p, ff[fault], k=k)
            feats[f"sb_{fault}_lo"] = lo
            feats[f"sb_{fault}_hi"] = hi
        feats[f"er_{fault}"] = fault_energy_share(f, p, ff[fault])
    return feats


def build_feature_set(metadata, index, save=True):
    """Per-window feature table for the whole corpus (MCSA-only).

    Rows: recording, bearing, condition, rep, label, window_id + one feature
    column per (channel, feature). Columns are prefixed pc1_/pc2_.
    """
    idx = index.set_index("recording")
    rows = []
    for rec_id, rec in idx.iterrows():
        path = rec["filepath"]
        signals = load_recording(path)
        speed_rpm = int(path.split("/")[-1].split("_")[0][1:]) * 100

        base = {"recording": rec_id,
                "bearing": rec["bearing"],
                "condition": rec["condition"],
                "rep": int(rec["rep"]),
                "label": rec["label"]}

        wins = {ch: list(sliding_windows(signals[ch], window=WINDOW, step=STEP))
                for ch in CURRENT_CHANNELS}
        if rec["n_windows"] != len(wins["phase_current_1"]):
            raise ValueError(
                f"{rec_id}: index says {rec['n_windows']} windows, got "
                f"{len(wins['phase_current_1'])}")

        for i in range(rec["n_windows"]):
            row = dict(base, window_id=i)
            for ch, prefix in CURRENT_CHANNELS.items():
                for feat, val in window_features(wins[ch][i], speed_rpm).items():
                    row[f"{prefix}_{feat}"] = val
            rows.append(row)

    feats = pd.DataFrame(rows)
    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        feats.to_csv(PROCESSED_DIR / "features_window.csv", index=False)
    return feats