"""Production pipeline configuration (Phase 2: train a real detector).

Carries the exploration decisions forward:
- MCSA-only: the two phase currents are the ONLY fault features; vibration is
  left out of the model (no vibration sensor in deployment).
- Option B: whole recordings are kept; fixed sliding windows absorb the
  variable recording length. No trimming anywhere.
- Bearing-wise splits: recordings of one bearing never straddle a fold or a
  partition; windowing happens strictly after the split assignment.

Production window size: 16384 samples @ 64 kHz = 256 ms with 50% overlap.
This yields 3.9 Hz frequency bins so the MCSA sidebands (|49 - d| = 26 Hz and
125.7 Hz) are resolved individually.
"""

from pathlib import Path

from utils.spectral import GEOMETRY, LINE_FREQ, RATE_CUR

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROCESSED_DIR / "results"
MODEL_DIR = DATA_DIR / "models"

# Windowing (Option B)
WINDOW = 16_384          # samples per window (256 ms @ 64 kHz)
STEP = 8_192             # 50% overlap
NPERSEG_FEAT = 16_384    # Welch nperseg == window -> 3.9 Hz bins

# Channels used as model features (MCSA scope)
CURRENT_CHANNELS = {"phase_current_1": "pc1", "phase_current_2": "pc2"}

# Fault-frequency families and how many supply-side harmonics to look at
FAULT_TYPES = ["bpfo", "bpfi", "bsf", "ftf"]
SIDEBAND_HARMONICS = [1]  # k in line_freq +/- k*fault_hz

# Deep (raw-signal) pipeline. The phase currents (64 kHz) are decimated by 16 ->
# 4 kHz sampling: motor-current MCSA content lives below ~150 Hz (line 50 Hz,
# sidebands 26-126 Hz), so nothing relevant is lost while CNN compute drops
# ~256x and the on-disk cache stays tiny. Windows stay 256 ms at 1024 samples.
DECIM = 16
DEEP_WINDOW = WINDOW // DECIM      # 1024 samples
DEEP_STEP = STEP // DECIM          # 512 samples
DEEP_DIR = PROCESSED_DIR / "deep"
DEEP_CACHE = DEEP_DIR / "window_cache.npy"      # float16 (N, 1024, 2)
DEEP_KEYS = DEEP_DIR / "window_cache_keys.csv"  # recording, window_id

# Keras backend: JAX (TensorFlow has no wheels for this Python; same Keras API).
KERAS_BACKEND = "jax"

SEED = 42