"""Anomaly detection & signal sanity scan (detect-only, no fixes).

Exploration deliverable: reports WHICH channels/recordings deviate from the
expected physical ranges. Deciding what to do about them happens in Cleaning.
"""

import numpy as np

from .loaders import load_recording
from .metadata import DATA_RAW, PROJECT_ROOT

# Expected physical limits per channel (units from CHANNEL_CATALOG).
# Exploration assumptions -- tuning happens in the Cleaning milestone.

LIMITS = {
    "vibration_1":           {"min": -10.0, "max": 10.0},
    "phase_current_1":       {"min": -20.0, "max": 20.0},
    "phase_current_2":       {"min": -20.0, "max": 20.0},
    "speed":                 {"min": 0.0,   "max": 2000.0},
    "torque":                {"min": 0.0,   "max": 2.0},
    "force":                 {"min": 0.0,   "max": 5000.0},
    "temp_2_bearing_module": {"min": 0.0,   "max": 150.0},
}

REPORT_FILE = PROJECT_ROOT / "data" / "processed" / "anomaly_report.csv"

def scan_signal_sanity(path):
    """Check one recording for signal anomalies.

    Returns {channel: {nan, inf, flat, n_unique, min, max, within_limits}}.
    Detection ONLY -- no data is modified.
    """
    signals = load_recording(path)
    report = {}
    for name, data in signals.items():
        row = {
            "filepath": str(path),
            "channel": name,
            "samples": data.size,
            "nan": int(np.isnan(data).sum()),
            "inf": int(np.isinf(data).sum()),
            "n_unique": int(np.unique(data).size),
            "flat": bool(np.unique(data).size <= 1),
            "min": float(np.nanmin(data)),
            "max": float(np.nanmax(data)),
            "within_limits": None,
        }

        limits = LIMITS.get(name)
        if limits is not None:
            row["within_limits"] = bool(
                np.nanmin(data) >= limits["min"] and np.nanmax(data) <= limits["max"]
            )
        report[name] = row
    return report

def scan_corpus(data_dir=DATA_RAW, max_files=None):
    """Scan recordings and aggregate a per-channel anomaly report.

    Returns a pandas DataFrame of facts. Optionally saves REPORT_FILE.
    """
    import pandas as pd

    rows = []
    files = sorted(data_dir.rglob("*.mat"))
    if max_files is not None:
        files = files[:max_files]

    for i, f in enumerate(files, 1):
        try:
            for row in scan_signal_sanity(f).values():
                rows.append(row)
        except Exception as exc:
            rows.append({
                "filepath": str(f), "channel": "",
                "samples": None, "nan": None, "inf": None,
                "n_unique": None, "flat": None, "min": None, "max": None,
                "within_limits": None,
                "error": f"{type(exc).__name__}: {exc}",
            })
        if i % 400 == 0:
            print(f"  scanned {i}/{len(files)}")

    return pd.DataFrame(rows)


def summarize_anomalies(df):
    """Condense the raw scan into a compact findings table."""
    return df.groupby("channel").agg(
        files=("filepath", "count"),
        nan_files=("nan", lambda s: (s > 0).sum()),
        inf_files=("inf", lambda s: (s > 0).sum()),
        flat_files=("flat", lambda s: (s == True).sum()),
        out_of_limits=("within_limits", lambda s: (s == False).sum()),
        min_value=("min", "min"),
        max_value=("max", "max"),
    )
def save_report(df, out_path=REPORT_FILE):
    """Write the anomaly report to CSV (explicit opt-in only)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"anomaly report saved: {out_path}")

