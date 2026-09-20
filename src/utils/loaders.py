import numpy as np
from scipy.io import loadmat


# Channel metadata registry: rate (Hz), samples per 4s recording, role, unit.
CHANNEL_CATALOG = {
    "vibration_1": {"rate": 64_000, "samples": 256_001, "role": "fault_primary", "unit": "g"},
    "phase_current_1": {"rate": 64_000, "samples": 256_001, "role": "fault_secondary", "unit": "A"},
    "phase_current_2": {"rate": 64_000, "samples": 256_001, "role": "fault_secondary", "unit": "A"},
    "speed": {"rate": 4_000, "samples": 16_001, "role": "condition", "unit": "rpm"},
    "torque": {"rate": 4_000, "samples": 16_001, "role": "condition", "unit": "Nm"},
    "force": {"rate": 4_000, "samples": 16_001, "role": "condition", "unit": "N"},
    "temp_2_bearing_module": {"rate": 1, "samples": 5, "role": "condition", "unit": "degC"},
}


def load_recording(path):
    """Read one Paderborn .mat file.

    Returns {channel_name: np.ndarray}, e.g.
    {'vibration_1': np.ndarray, 'phase_current_1': np.ndarray, ...}
    """
    mat = loadmat(str(path))
    key = [k for k in mat.keys() if not k.startswith("__")][0]
    rec = mat[key][0, 0]

    return {
        str(np.ravel(rec["Y"][0, j]["Name"])[0]): np.ravel(rec["Y"][0, j]["Data"])
        for j in range(rec["Y"].shape[1])
    }


def load_currents(metadata, bearing, sub):
    """Both phase currents at a given operating-condition substring."""
    path = metadata[(metadata["bearing"] == bearing) & metadata["filepath"].str.contains(sub)][
        "filepath"
    ].iloc[0]
    s = load_recording(path)
    return s["phase_current_1"], s["phase_current_2"]


def inspect_recordings(path):
    """Load a recording and return per-channel schema + value stats.

    Use it to confirm a file matches the expected channel catalog.
    Returns {channel: {'shape', 'dtype', 'min', 'max', 'samples', 'expected_samples'}}.
    """

    signals = load_recording(path)
    reports = {}
    for name, data in signals.items():
        row = {
            "shape": data.shape,
            "dtype": str(data.dtype),
            "samples": data.size,
            "min": float(np.min(data)),
            "max": float(np.max(data)),
        }
        if name in CHANNEL_CATALOG:
            expected = CHANNEL_CATALOG[name]["samples"]
            row["expected_samples"] = expected
            row["matches_catalog"] = data.size == expected
        else:
            row["expected_samples"] = None
            row["matches_catalog"] = None
        reports[name] = row
    return reports
