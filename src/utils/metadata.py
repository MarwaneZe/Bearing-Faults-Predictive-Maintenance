from pathlib import Path

import pandas as pd

from .parsers import get_label, parse_filename

DATA_RAW = Path("../data/raw")
METADATA_FILE = Path("../data/processed/metadata.csv")


def build_metadata(data_dir=DATA_RAW, metadata_file=METADATA_FILE, refresh=False):
    """Build a metadata DataFrame from all .mat files and save as CSV.

    Columns: filepath, bearing, speed_rpm, torque_Nm, force_N, rep, label

    If the CSV already exists and refresh=False, load it instead of
    rescanning the raw data.
    """
    data_dir = Path(data_dir)
    metadata_file = Path(metadata_file)

    if data_dir == DATA_RAW and metadata_file.exists() and not refresh:
        return pd.read_csv(metadata_file)

    rows = []
    for file_path in sorted(data_dir.rglob("*.mat")):
        info = parse_filename(file_path)
        info["filepath"] = str(file_path)
        info["label"] = get_label(info["bearing"])
        rows.append(info)

    df = pd.DataFrame.from_records(rows)

    if data_dir == DATA_RAW:
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(metadata_file, index=False)

    return df
