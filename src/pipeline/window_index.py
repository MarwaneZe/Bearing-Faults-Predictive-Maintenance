"""Cleaning : split-tagged window index.

Materializes Option B: for every usable recording it computes the number of
fixed windows the signal yields and tags the recording with its bearing-wise
Split A fold and Split B partition. Windowing "after the split" is guaranteed
because the tag lives at recording level and every window inherits it.

Output: data/processed/window_index.csv  (one row per recording).
"""

import pandas as pd

from utils.loaders import load_recording
from utils.splits import assign_folds, assignment_b, verify
from pipeline.config import PROCESSED_DIR, WINDOW, STEP


def _window_count(n_samples, window=WINDOW, step=STEP):
    """Number of full windows for a signal of n_samples (Option B)."""
    if n_samples < window:
        return 0
    return (n_samples - window) // step + 1


def build_window_index(metadata, save=True):
    """Return metadata + {n_windows, fold, partition, condition, recording}."""
    meta = metadata.copy()
    meta["condition"] = meta.apply(
        lambda r: f"N{int(r.speed_rpm / 100):02d}_M{int(r.torque_Nm * 10):02d}"
                  f"_F{int(r.force_N / 100):02d}",
        axis=1,
    )
    meta["recording"] = meta["filepath"].apply(lambda p: p.split("/")[-1])

    meta["fold"] = assign_folds(meta, seed=42)["fold"]
    meta["partition"] = assignment_b(meta)["partition"]

    n = []
    for _, rec in meta.iterrows():
        n.append(_window_count(load_recording(rec["filepath"])["phase_current_1"].size))
    meta["n_windows"] = n

    verify(meta, meta, "A")
    verify(meta, meta, "B")

    cols = ["recording", "bearing", "condition", "rep", "label",
            "n_windows", "fold", "partition", "filepath"]
    index = meta[cols].reset_index(drop=True)
    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        index.to_csv(PROCESSED_DIR / "window_index.csv", index=False)
    return index