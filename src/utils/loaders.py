import numpy as np
from scipy.io import loadmat


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
    path = metadata[
        (metadata["bearing"] == bearing)
        & metadata["filepath"].str.contains(sub)
    ]["filepath"].iloc[0]
    s = load_recording(path)
    return s["phase_current_1"], s["phase_current_2"]
