"""Shared orchestration for the two experiments.

Experiment 1 (detector): hold out 6 bearings, compare 6 independent models by
bearing-wise CV over the rest, rank, fit the winner, tune a threshold, demo on
the held-out files.
Experiment 2 (transfer): train every model on artificial+healthy only, evaluate
ONCE on real/combination (+ healthy reference), no cherry-picking.
"""

import numpy as np
import pandas as pd

from models import zoo
from pipeline.config import PROCESSED_DIR, SEED
from utils.loaders import load_recording
from utils.metrics import (
    bearing_level,
    bearing_scores,
    platt_scale,
    threshold_for_fpr,
    threshold_scores,
    to_int,
)

ALL_MODELS = zoo.ALL_MODELS


def feature_cols(df):
    return [c for c in df.columns if c.startswith("pc1_") or c.startswith("pc2_")]


def load_data():
    """Windows + engineered features + split tags (one row per window)."""
    feats = pd.read_csv(PROCESSED_DIR / "features_window.csv")
    idx = pd.read_csv(PROCESSED_DIR / "window_index.csv")
    df = feats.merge(idx[["recording", "fold", "partition"]], on="recording", how="left")
    return df, df[feature_cols(df)].to_numpy()


def load_cache():
    from deep.models import _positions
    from deep.windows import build_cache

    cache, _ = build_cache()
    return cache


def run_cv(df, X, cache, name, seed, n_folds=5, train_stride=2, verbose=False):
    """Bearing-wise CV of ONE model for ONE seed. Returns rows/oof/scores."""
    if name in zoo.FEATURE_MODELS:
        return zoo.eval_cv(df, X, name, seed, n_folds)
    if name == "mlp":
        from deep.models import eval_cv_features

        return eval_cv_features(df, X, name, seed, n_folds)
    if name == "cnn":
        from deep.models import eval_cv_cache

        return eval_cv_cache(df, cache, name, seed, n_folds, train_stride)
    raise ValueError(name)


def cv_summary(df, X, cache, seeds=(SEED, 7), n_folds=5, train_stride=2, verbose=True):
    """Rank ALL models on bearing-level metrics, mean +/- std over seeds."""
    feats = feature_cols(df)
    rows, oof_store = [], {}
    metrics_cols = ["window_auroc", "window_aupr", "bearing_auroc", "bearing_aupr",
                    "auroc_N15_M07_F10", "auroc_N09_M07_F10",
                    "auroc_N15_M01_F10", "auroc_N15_M07_F04"]
    for name in ALL_MODELS:
        per_seed = []
        oof_seeds = []
        for seed in seeds:
            res = run_cv(df, X, cache, name, seed, n_folds, train_stride)
            fold = res["rows"]
            per_seed.append({
                **res["bearing_scores"],
                "window_auroc": float(fold["window_auroc"].mean()),
                "window_aupr": float(fold["window_aupr"].mean()),
            })
            oof_seeds.append(res["oof"])
        data = {c: [s[c] for s in per_seed] for c in metrics_cols}
        summary = {
            "model": name,
            **{f"{c}±": float(np.round(np.nanmean(v), 3)) for c, v in data.items()},
            **{f"{c}±std": float(np.round(np.nanstd(v), 3)) for c, v in data.items()},
        }
        rows.append(summary)
        y = np.concatenate([o[0] for o in oof_seeds])
        p = np.concatenate([o[1] for o in oof_seeds])
        oof_store[name] = (y, p)
    table = pd.DataFrame(rows)
    if verbose:
        show = ["model", "bearing_auroc±", "bearing_auroc±std", "window_auroc±",
                "auroc_N15_M01_F10±", "auroc_N09_M07_F10±"]
        print(table[show].to_string(index=False))
    return table, oof_store


def best_model(cv_table, metric="bearing_auroc±"):
    ranked = cv_table.sort_values(metric, ascending=False, na_position="last")
    return ranked.iloc[0]["model"], ranked


def fit_final(df_train, X, cache, name, seed=SEED, n_folds=5, train_stride=2):
    """Fit ONE model on all training bearings. Returns (predictor, kind).

    Keras nets get a bearing-wise early-stopping validation subset carved out of
    df_train (fold 0 of a fresh assignment); sklearn models fit directly.
    """
    if name in zoo.FEATURE_MODELS:
        model = zoo.build_model(name, to_int(df_train["label"]), X.shape[1])
        model.fit(X, to_int(df_train["label"]))
        return model, "sklearn"

    from deep.models import _positions, fit_net
    from utils.splits import assign_folds_on

    df = assign_folds_on(df_train.reset_index(drop=True), n_folds=n_folds, seed=seed)
    val_mask = df["fold"] == 0
    tr, va = ~val_mask, val_mask
    if name == "mlp":
        Xtr, Xva = X[tr.to_numpy()], X[va.to_numpy()]
    else:  # cnn
        pos = _positions(df)
        Xtr = np.asarray(cache[pos[tr.to_numpy()]], dtype=np.float32)
        Xva = np.asarray(cache[pos[va.to_numpy()]], dtype=np.float32)
    model = fit_net(Xtr, df["label"].to_numpy()[tr.to_numpy()],
                    Xva, df["label"].to_numpy()[va.to_numpy()], name, seed)
    return model, "keras"


def predict(df_hit, X, cache, model, name):
    """P(damaged) for each window of df_hit."""
    if name in zoo.FEATURE_MODELS or name == "mlp":
        return model.predict_proba(X)[:, 1] if name in zoo.FEATURE_MODELS else model.predict(X, verbose=0).ravel()
    from deep.models import _positions

    pos = _positions(df_hit)
    Xh = np.asarray(cache[pos], dtype=np.float32)
    return model.predict(Xh, verbose=0).ravel()


def recording_verdicts(df_hit, prob, threshold):
    """Per-recording mean damage probability -> verdict (the demo table)."""
    out = df_hit[["recording", "bearing", "condition", "label"]].copy()
    out["p_damaged"] = np.asarray(prob)
    agg = out.groupby("recording", as_index=False).agg(
        bearing=("bearing", "first"), condition=("condition", "first"),
        label=("label", "first"), p_damaged=("p_damaged", "mean"),
        n_windows=("p_damaged", "size"),
    )
    agg["verdict"] = (agg["p_damaged"] >= threshold).map({True: "damaged", False: "healthy"})
    return agg.sort_values("p_damaged", ascending=False, kind="stable")


def calibrate_and_threshold(y_oof, p_oof, y_new, p_new, max_fpr=0.10):
    """Platt-calibrate on out-of-fold probs, threshold for <= max_fpr FPR."""
    p_cal = platt_scale(y_oof, p_oof, p_new)
    t = threshold_for_fpr(y_oof, p_oof, max_fpr=max_fpr)
    return p_cal, t


def transfer_full(df_tr, X_tr, df_tt, X_tt, cache, names=ALL_MODELS, max_fpr=0.10):
    """Experiment 2: train each model on artificial+healthy, evaluate on real.

    Returns (all-models table, per-model verdict metrics at a threshold tuned
    on an artificial-only OOF CV, per-model calibrated test probabilities).
    """
    from utils.metrics import auroc, aupr

    table_rows, ver_rows, probs = [], [], []
    for name in names:
        y_oof, p_oof = run_cv(df_tr, X_tr, cache, name, seed=SEED, n_folds=5)["oof"]
        model, _ = fit_final(df_tr, X_tr, cache, name)
        p = predict(df_tt, X_tt, cache, model, name)
        y = to_int(df_tt["label"])
        bl = bearing_level(df_tt, p)
        table_rows.append({
            "model": name,
            "window_auroc": auroc(y, p), "window_aupr": aupr(y, p),
            **bearing_scores(bl),
        })
        t = threshold_for_fpr(y_oof, p_oof, max_fpr=max_fpr)
        p_cal = platt_scale(y_oof, p_oof, p)
        ver_rows.append({"model": name, **threshold_scores(df_tt["label"], p_cal, t)})
        probs.append((name, p_cal))
    return pd.DataFrame(table_rows), pd.DataFrame(ver_rows), probs