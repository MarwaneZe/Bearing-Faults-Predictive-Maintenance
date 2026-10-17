"""Raw-window cache for the deep models (Keras, JAX backend).

Materializes every 256 ms current window ONCE as a float16 memmap aligned with
`window_index.csv` (same recording order), then decimates by 8 and L2-normalises
per window/channel. The split tags (fold / partition) live in window_index, so
the cache itself knows nothing about splits - bearing-wise partitions are
reproduced exactly by filtering on the recording.

Cache: data/processed/deep/window_cache.npy   shape (N_windows, 2048, 2)
        data/processed/deep/window_cache_keys.csv  (recording, window_id)
"""

import numpy as np
import pandas as pd

from pipeline.config import (
    DEEP_CACHE,
    DEEP_DIR,
    DEEP_KEYS,
    DEEP_STEP,
    DEEP_WINDOW,
    DECIM,
    PROCESSED_DIR,
)
from utils.loaders import load_recording

WINDOW_SRC = DEEP_WINDOW * DECIM   # 16384 samples @64 kHz
STEP_SRC = DEEP_STEP * DECIM       # 8192 samples @64 kHz


def _decimate(signal, factor=DECIM):
    return np.asarray(signal, float)[::factor]


def _normalize_window(w):
    """Per-channel mean-centre + unit-RMS (amplitude-scale invariant)."""
    w = np.asarray(w, float)
    mu = w.mean(axis=0, keepdims=True)
    sd = w.std(axis=0, keepdims=True)
    return ((w - mu) / (sd + 1e-6)).astype(np.float16)


def build_cache(force=False):
    """(Re)build the decimated window cache from raw .mat recordings.

    Returns (cache_memmap, keys_df). Verifies the window count matches
    window_index.csv exactly.
    """
    index = pd.read_csv(PROCESSED_DIR / "window_index.csv")
    if DEEP_CACHE.exists() and DEEP_KEYS.exists() and not force:
        return np.load(DEEP_CACHE, mmap_mode="r"), pd.read_csv(DEEP_KEYS)

    total = int(index["n_windows"].sum())
    DEEP_DIR.mkdir(parents=True, exist_ok=True)
    cache = np.lib.format.open_memmap(DEEP_DIR / "window_cache.npy", mode="w+",
                                      dtype=np.float16, shape=(total, DEEP_WINDOW, 2))
    keys = []
    row = 0
    for _, rec in index.iterrows():
        sig = load_recording(rec["filepath"])
        p1_full = np.asarray(sig["phase_current_1"], float)
        p2_full = np.asarray(sig["phase_current_2"], float)
        n = (p1_full.size - WINDOW_SRC) // STEP_SRC + 1 if p1_full.size >= WINDOW_SRC else 0
        if n != rec["n_windows"]:
            raise ValueError(
                f"{rec['recording']}: replay gives {n} windows, index says {rec['n_windows']}")
        p1 = p1_full[::DECIM]
        p2 = p2_full[::DECIM]
        for i, start in enumerate(range(0, (n - 0) * DEEP_STEP, DEEP_STEP)):
            w = np.stack([p1[start:start + DEEP_WINDOW], p2[start:start + DEEP_WINDOW]], axis=1)
            cache[row] = _normalize_window(w)
            keys.append((rec["recording"], i))
            row += 1
    cache.flush()
    pd.DataFrame(keys, columns=["recording", "window_id"]).to_csv(DEEP_KEYS, index=False)
    print(f"deep cache built: {row} windows -> {DEEP_CACHE}")
    return np.load(DEEP_CACHE, mmap_mode="r"), pd.read_csv(DEEP_KEYS)


def window_row_ids(records):
    """Positional indices into the cache for a set of recording names."""
    keys = pd.read_csv(DEEP_KEYS)
    return keys.index[keys["recording"].isin(records)].to_numpy()


def cache_slice(cache, idx, dtype=np.float32):
    """Subset of the cache as a dense float32 (N, 2048, 2) array."""
    return np.asarray(cache[idx], dtype=dtype)