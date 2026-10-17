"""Time-domain signal features & class-separation sanity check.

Exploration principle: DETECT-ONLY. Trimming is done ON-THE-FLY (in memory)
for exploration; materialized cleaning happens in a later milestone.
"""

import numpy as np
from scipy.stats import skew, kurtosis

from .loaders import load_recording
from .parsers import condition_key, get_label

# Exploration defaults for windowing (production config: src/pipeline/config.py).
WINDOW = 4096  # samples @64kHz = 64 ms
STEP = 2048  # 50% overlap

FAULT_CHANNELS = ["vibration_1", "phase_current_1", "phase_current_2"]


def time_domain_features(signal):
    """Statistical signature of one signal (full length, no trimming).

    Returns {rms, mean, std, peak, peak_to_peak, crest_factor, skewness, kurtosis}.
    """
    x = np.asarray(signal, dtype=float)
    if x.size == 0:
        return {"rms": np.nan, "mean": np.nan, "std": np.nan, "peak": np.nan,
                "peak_to_peak": np.nan, "crest_factor": np.nan,
                "skewness": np.nan, "kurtosis": np.nan}
    rms = float(np.sqrt(np.mean(x ** 2)))
    amp_peak = float(np.max(np.abs(x)))
    return {
        "rms": rms,
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "peak": amp_peak,
        "peak_to_peak": float(np.max(x) - np.min(x)),
        "crest_factor": float(amp_peak / rms) if rms > 0 else 0.0,
        "skewness": float(skew(x)),
        "kurtosis": float(kurtosis(x)),
    }


def sliding_windows(signal, window=WINDOW, step=STEP):
    """Yield fixed-length windows of one signal (Option B).

    A trailing partial window (length < window) is dropped.
    """
    x = np.asarray(signal, dtype=float)
    for start in range(0, len(x) - window + 1, step):
        yield x[start:start + window]


def build_feature_table(metadata, channels=None, bearings=None, reps=None):
    """Per-recording time-domain features for a sample of the corpus.

    Stats are computed on FULL-length signals; true length is recorded in the
    `samples` column. Returns DataFrame: bearing, condition, rep, label,
    samples + one column per (channel, feature) pair.
    """
    import pandas as pd

    if channels is None:
        channels = FAULT_CHANNELS

    sub = metadata
    if bearings is not None:
        sub = sub[sub["bearing"].isin(bearings)]
    if reps is not None:
        sub = sub[sub["rep"].isin(reps)]

    rows = []
    for _, rec in sub.iterrows():
        try:
            signals = load_recording(rec["filepath"])
        except Exception:
            continue  # corrupt/missing files already documented in Blocks 1/4
        row = {
            "bearing": rec["bearing"],
            "condition": condition_key(rec),
            "rep": rec["rep"],
            "label": get_label(rec["bearing"]),
        }
        for ch in channels:
            if ch in signals:
                row[f"{ch}_samples"] = signals[ch].size
                for feat, val in time_domain_features(signals[ch]).items():
                    row[f"{ch}_{feat}"] = val
        rows.append(row)
    return pd.DataFrame(rows)