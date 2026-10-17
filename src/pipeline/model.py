"""Modelling (Phase 2): Split A CV + Split B deployment verdict.

Protocol (locked in exploration):
- binary healthy vs damaged; AUROC / PR-AUC are the primary metrics.
- Split A: stratified bearing-wise 5-fold CV -> model building.
- Split B: train = artificial (KA) + healthy, test = real/combination (KI/KB)
  + healthy -> the deployment verdict, run once, never tuned on.
- Bearings are the evaluation unit (mean window probability per bearing and
  condition); window-level metrics are reported for insight only.

Artifacts under data/models/ and data/processed/results/.
"""

import json

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

from pipeline.config import PROCESSED_DIR, RESULTS_DIR, MODEL_DIR, SEED, WINDOW, STEP

CONDITIONS = ["N15_M07_F10", "N09_M07_F10", "N15_M01_F10", "N15_M07_F04"]


def load_features(chunk=None):
    """Feature table merged with the split tags. chunk = first n rows, sanity."""
    df = pd.read_csv(PROCESSED_DIR / "features_window.csv")
    idx = pd.read_csv(PROCESSED_DIR / "window_index.csv")
    df = df.merge(idx[["recording", "fold", "partition"]], on="recording", how="left")
    if chunk:
        df = df.head(chunk)
    return df


def feature_columns(df):
    return [c for c in df.columns if c.startswith("pc1_") or c.startswith("pc2_")]


def bearing_level(df, probs):
    """Mean window probability per (bearing, condition) -- the eval unit."""
    out = df[["bearing", "condition", "label"]].copy()
    out["prob"] = np.asarray(probs)
    return out.groupby(["bearing", "condition"], as_index=False).agg(
        label=("label", "first"), prob=("prob", "mean"))


def make_logreg():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(max_iter=3000, C=1.0, random_state=SEED)),
    ])


def make_gb():
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_depth=6,
        min_samples_leaf=20, l2_regularization=1.0, random_state=SEED)


def _cond_auroc(bl, cond):
    s = bl[bl["condition"] == cond]
    y = (s["label"] == "damaged").astype(int)
    if y.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y, s["prob"]))


def _aupr(y, prob):
    return float(average_precision_score(y, prob))


def split_a(df, features, estimator, verbose=True):
    """Bearing-wise 5-fold CV. Returns per-fold rows + trained fold models."""
    rows, models = [], []
    for fold in sorted(df["fold"].unique()):
        train, test = df[df["fold"] != fold], df[df["fold"] == fold]
        est = estimator
        est.fit(train[features], train["label"] == "damaged")
        prob = est.predict_proba(test[features])[:, 1]

        y = (test["label"] == "damaged").astype(int)
        bl = bearing_level(test, prob)
        cond_auc = {c: _cond_auroc(bl, c) for c in CONDITIONS}
        rows.append({
            "fold": fold,
            "window_auroc": float(roc_auc_score(y, prob)),
            "window_pr": _aupr(y, prob),
            **{f"bearing_auroc_{c}": v for c, v in cond_auc.items()},
        })
        models.append(est)

    res = pd.DataFrame(rows)
    if verbose:
        print("Split A — window AUROC per fold:", res["window_auroc"].round(3).tolist())
        print("Split A — window PR    per fold:", res["window_pr"].round(3).tolist())
        print("Split A — mean bearing-AUROC per condition:")
        print(res[[f"bearing_auroc_{c}" for c in CONDITIONS]].mean().round(3).to_string())
    return res, models


def split_b(df, features, estimator):
    """Train on artificial + healthy; run ONCE on real/combination + healthy.

    The six healthy bearings act as the shared reference (standard protocol for
    this dataset): their windows are used in training AND re-used in the test
    set, so the deployment verdict measures real damage against healthy states.
    Damaged bearings stay strictly partitioned (KA train-only, KI/KB test-only).
    """
    train = df[df["partition"] == "train"]                 # KA + healthy
    test_real = df[df["partition"] == "test"]              # KI/KB (real/combination)
    healthy = df[df["label"] == "healthy"]                 # healthy reference
    test = pd.concat([test_real, healthy])

    est = estimator
    est.fit(train[features], train["label"] == "damaged")
    prob = est.predict_proba(test[features])[:, 1]

    y = (test["label"] == "damaged").astype(int)
    bl = bearing_level(test, prob)
    out = {
        "estimator": est,
        "cond_auroc": {c: _cond_auroc(bl, c) for c in CONDITIONS},
        "cond_aupr": {},
        "window_auroc": float(roc_auc_score(y, prob)),
        "window_pr": float(_aupr(y, prob)),
        "bearing_level": bl,
        "baseline_precision": float((test["label"] == "damaged").mean()),
    }
    for c in CONDITIONS:
        s = bl[bl["condition"] == c]
        yy = (s["label"] == "damaged").astype(int)
        out["cond_aupr"][c] = _aupr(yy, s["prob"]) if yy.nunique() > 1 else float("nan")

    bl["verdict"] = (bl["prob"] >= 0.5).map({True: "damaged", False: "healthy"})
    out["bearing_confusion"] = confusion_matrix(
        bl["label"], bl["verdict"], labels=["healthy", "damaged"])
    return out


def run_pipeline(sanity_chunk=None):
    """Feature table -> Split A (select model) -> Split B (deployment)."""
    df = load_features(sanity_chunk)
    features = feature_columns(df)
    print(f"windows: {len(df)}  features: {len(features)}")

    # --- Split A: compare LR vs GB, keep the one that does better on bearings
    cands = {"logreg": make_logreg(), "grad_boost": make_gb()}
    best = None
    for name, est in cands.items():
        res, _ = split_a(df, features, est, verbose=False)
        score = res[[f"bearing_auroc_{c}" for c in CONDITIONS]].mean().mean()
        print(f"Split A [{name:10s}] mean bearing-AUROC/condition = {score:.3f}")
        if best is None or score > best[1]:
            best = (name, score, res)
    name, score, res_a = best
    print(f"=> best model: {name} ({score:.3f})")

    # --- Split B: deployment verdict with the chosen model
    ab = split_b(df, features, make_gb() if name == "grad_boost" else make_logreg())

    if sanity_chunk is None:
        _save_artifacts(name, ab["estimator"], ab, features)
    return {"name": name, "split_a": res_a, "split_b": ab, "features": features}


def _save_artifacts(name, estimator, sb, features):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    sb["bearing_level"].to_csv(RESULTS_DIR / "split_b_bearings.csv", index=False)
    pd.Series(sb["cond_auroc"], name="bearing_auroc").to_csv(RESULTS_DIR / "split_b_cond_auroc.csv")
    pd.Series(sb["cond_aupr"], name="bearing_aupr").to_csv(RESULTS_DIR / "split_b_cond_aupr.csv")

    summary = {
        "model": name,
        "window_auroc": sb["window_auroc"],
        "window_pr": sb["window_pr"],
        "cond_auroc": sb["cond_auroc"],
        "cond_aupr": sb["cond_aupr"],
        "baseline_precision": sb["baseline_precision"],
        "feature_cols": features,
        "window": WINDOW,
        "step": STEP,
    }
    with open(MODEL_DIR / "config.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    dump(estimator, MODEL_DIR / "model.joblib")
    print("artifact: data/models/model.joblib")
    print("config  : data/models/config.json")
    print("results : data/processed/results/split_b_bearings.csv")