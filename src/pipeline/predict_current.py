"""Deployment predictor: live current samples -> fault verdict.

Usage (CLI):
    python -m src.pipeline.predict_current --mat <paderborn.mat> [--speed 1500]
    python -m src.pipeline.predict_current --npz <cur1.npy> <cur2.npy> [--speed 1500]

API:
    from src.pipeline.predict_current import load_model, predict_stream
    model, cfg = load_model()
    df = predict_stream(phase1, phase2, speed_rpm=1500, model=model, cfg=cfg)

The verdict threshold is 0.5 on the mean window probability.
"""

import argparse
import json

import numpy as np
import pandas as pd
from joblib import load

from utils.loaders import load_recording
from utils.features import sliding_windows
from pipeline.config import MODEL_DIR


def load_model(model_file=None, cfg_file=None):
    model_file = model_file or MODEL_DIR / "model.joblib"
    cfg_file = cfg_file or MODEL_DIR / "config.json"
    with open(cfg_file) as fh:
        cfg = json.load(fh)
    return load(model_file), cfg


def predict_stream(phase1, phase2, speed_rpm, model=None, cfg=None):
    """Probability of 'damaged' per 256 ms window of the two phase currents."""
    model, cfg = model or load_model()[0], cfg or load_model()[1]
    from pipeline.features import window_features  # local import: keep CLI light

    feats_col = cfg["feature_cols"]
    windows_1 = list(sliding_windows(np.asarray(phase1, float), cfg["window"], cfg["step"]))
    windows_2 = list(sliding_windows(np.asarray(phase2, float), cfg["window"], cfg["step"]))

    rows = []
    for i, (w1, w2) in enumerate(zip(windows_1, windows_2)):
        row = {}
        for ch_name, w in (("phase_current_1", w1), ("phase_current_2", w2)):
            for feat, val in window_features(w, speed_rpm).items():
                row[f"pc1_{feat}" if ch_name == "phase_current_1" else f"pc2_{feat}"] = val
        rows.append(row)

    pred = pd.DataFrame(rows)[feats_col]
    prob = model.predict_proba(pred)[:, 1]
    return pd.DataFrame({"window_id": np.arange(len(windows_1)), "p_damaged": prob})


def main(argv=None):
    ap = argparse.ArgumentParser(description="MCSA bearing-fault predictor")
    ap.add_argument("--mat", help="Paderborn .mat recording")
    ap.add_argument("--npz", nargs=2, metavar=("CUR1", "CUR2"), help="two .npy current arrays")
    ap.add_argument("--speed", type=int, default=1500, help="shaft speed in rpm")
    args = ap.parse_args(argv)

    if args.mat:
        s = load_recording(args.mat)
        p1, p2 = s["phase_current_1"], s["phase_current_2"]
    elif args.npz:
        p1, p2 = (np.load(f) for f in args.npz)
    else:
        ap.error("provide --mat or --npz")

    df = predict_stream(p1, p2, args.speed)
    verdict = "damaged" if df["p_damaged"].mean() >= 0.5 else "healthy"
    print(df.describe().round(3).to_string())
    print(f"\nwindows: {len(df)} | mean p_damaged = {df['p_damaged'].mean():.3f}")
    print(f"verdict : {verdict}")
    return verdict


if __name__ == "__main__":
    main()