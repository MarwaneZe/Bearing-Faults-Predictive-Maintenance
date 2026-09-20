"""integrity audit: verify the raw dataset grid is complete and loadable."""

import re
from pathlib import Path
import pandas as pd

from .loaders import load_recording
from .metadata import PROJECT_ROOT, DATA_RAW

CONDITIONS = ["N15_M07_F10", "N09_M07_F10", "N15_M01_F10", "N15_M07_F04"]
REPS = list(range(1, 21))
BEARING_PATTERN = re.compile(r"^K[A-Z]{0,2}\d{2,3}$")
REPORT_FILE = PROJECT_ROOT / "data" / "processed" / "audit.csv"


def _discover_bearings(data_dir):
    """Return sorted bearing ids by scanning folder names in data/raw."""
    return sorted(
        p.name for p in Path(data_dir).iterdir() if p.is_dir() and BEARING_PATTERN.match(p.name)
    )


def audit_data_folder(data_dir=DATA_RAW, verify_loads=False):
    """Scan the whole data folder and report integrity.

    verify_loads=True actually loads every .mat file (slow, one-time).
    Returns a dict:
      {
        'bearings': [ids],
        'expected_files': int,
        'present_files': int,
        'missing': [(bearing, condition, rep)],
        'extra_files': [...],
        'unparseable': [...],
        'load_failures': [(filepath, error_type, message)],
        'channel_sets': {channel_tuple: count},
        'per_bearing': {bearing: {'present': n, 'expected': n, 'missing': [key...]}}
      }
    """

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {data_dir}")

    bearings = _discover_bearings(data_dir)
    expected_keys = {(b, c, r) for b in bearings for c in CONDITIONS for r in REPS}

    present = {}
    duplicates = []
    unparseable = []
    load_failures = []
    channel_sets = {}

    files = sorted(data_dir.rglob("*.mat"))
    for f in files:
        try:
            parts = f.stem.split("_")
            condition = "_".join(parts[:3])
            bearing = parts[3]
            rep = int(parts[4])
            key = (bearing, condition, rep)
        except (IndexError, ValueError):
            unparseable.append(str(f))
            continue

        if key in present:
            duplicates.append((key, str(f)))
            continue
        present[key] = str(f)

        if verify_loads:
            try:
                s = load_recording(f)
                chans = tuple(sorted(s.keys()))
                channel_sets[chans] = channel_sets.get(chans, 0) + 1
            except Exception as exc:
                load_failures.append((str(f), type(exc).__name__, str(exc)))

    missing = sorted(expected_keys - set(present.keys()))
    extra = sorted(set(present.keys()) - expected_keys)

    per_bearing = {}
    for b in bearings:
        expected_b = {(b, c, r) for c in CONDITIONS for r in REPS}
        missing_b = expected_b - set(present.keys())
        per_bearing[b] = {
            "present": len(expected_b) - len(missing_b),
            "expected": len(expected_b),
            "missing": sorted(missing_b),
        }

    report = {
        "bearings": bearings,
        "expected_files": len(expected_keys),
        "present_files": len(present),
        "missing": missing,
        "extra_files": extra,
        "duplicates": duplicates,
        "unparseable": unparseable,
        "load_failures": load_failures,
        "channel_sets": channel_sets,
        "per_bearing": per_bearing,
    }

    _save_report(report)

    if missing:
        print(f"[audit] MISSING {len(missing)} expected recordings")
    if extra:
        print(f"[audit] {len(extra)} unexpected (extra) recordings")
    if duplicates:
        print(f"[audit] {len(duplicates)} duplicate recordings")
    if unparseable:
        print(f"[audit] {len(unparseable)} unparseable filenames")
    if load_failures:
        print(f"[audit] {len(load_failures)} files failed to load")
    if len(channel_sets) > 1:
        print(f"[audit] WARNING: {len(channel_sets)} different channel sets found")
    if not missing and not extra and not duplicates and not load_failures:
        print("[audit] corpus is COMPLETE and CONSISTENT")

    return report


def _save_report(report):
    """Flatten issues into a CSV for the reports folder."""
    rows = []
    for bearing, cond, rep in report["missing"]:
        rows.append({"type": "missing", "bearing": bearing, "condition": cond, "rep": rep})
    for key, path in report["duplicates"]:
        rows.append(
            {
                "type": "duplicate",
                "bearing": key[0],
                "condition": key[1],
                "rep": key[2],
                "filepath": path,
            }
        )
    for path in report["unparseable"]:
        rows.append({"type": "unparseable", "filepath": path})
    for path, etype, msg in report["load_failures"]:
        rows.append({"type": "load_failure", "filepath": path, "error_type": etype, "message": msg})

    if not rows:
        return

    df = pd.DataFrame(rows)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORT_FILE, index=False)
