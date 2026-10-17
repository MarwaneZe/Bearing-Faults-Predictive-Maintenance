"""Shared evaluation metrics for both experiments (detector CV + transfer).

Rank metrics (AUROC / PR-AUC) are calibration-free and drive model ranking.
Threshold metrics (precision / recall / specificity / F1) are computed at an
operating point tuned on out-of-fold predictions ONLY -- never on the held-out
demo set (Experiment 1) nor on the real-damage test (Experiment 2).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

CONDITIONS = ["N15_M07_F10", "N09_M07_F10", "N15_M01_F10", "N15_M07_F04"]


def to_int(label):
    """label column -> 0/1 int array (healthy=0, damaged=1)."""
    return (np.asarray(label) == "damaged").astype(int)


def auroc(y, p):
    if np.unique(y).size < 2:
        return float("nan")
    p = np.asarray(p)
    if p.max() == p.min():  # degenerate/constant model
        return 0.5
    return float(roc_auc_score(y, p))


def aupr(y, p):
    if np.unique(y).size < 2:
        return float("nan")
    p = np.asarray(p)
    if p.max() == p.min():
        return float(np.asarray(y).mean())  # constant score == predicting the prior
    return float(average_precision_score(y, p))


def bearing_level(df, p):
    """Mean window probability per (bearing, condition) -- the evaluation unit."""
    out = df[["bearing", "condition", "label"]].copy()
    out["prob"] = np.asarray(p)
    return (
        out.groupby(["bearing", "condition"], as_index=False)
        .agg(label=("label", "first"), prob=("prob", "mean"))
    )


def bearing_scores(bl, p=None):
    """Per-condition bearing-level AUROC / PR-AUC (+ overall)."""
    if p is not None:
        bl = bl.copy()
        bl["prob"] = np.asarray(p)
    y = to_int(bl["label"].values)
    res = {"bearing_auroc": auroc(y, bl["prob"]), "bearing_aupr": aupr(y, bl["prob"])}
    for c in CONDITIONS:
        s = bl[bl["condition"] == c]
        res[f"auroc_{c}"] = auroc(to_int(s["label"]), s["prob"])
        res[f"aupr_{c}"] = aupr(to_int(s["label"]), s["prob"])
    return res


def threshold_for_fpr(y, p, max_fpr=0.10):
    """Highest threshold keeping >= (1 - max_fpr) of healthy above it.

    I.e. at threshold t, at most max_fpr of the HEALTHY windows are wrongly
    flagged as damaged. Fit on out-of-fold / calibration predictions only.
    """
    y = np.asarray(y)
    p = np.asarray(p)
    healthy = p[y == 0]
    if healthy.size == 0:
        return 0.5
    return float(np.quantile(healthy, 1.0 - max_fpr, method="higher"))


def threshold_scores(y, p, t):
    """Precision / recall / specificity / F1 / accuracy at threshold t."""
    y = to_int(y)
    p = np.asarray(p)
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(t),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": float(tp / (tp + fp)) if tp + fp else float("nan"),
        "recall": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else float("nan"),
        "accuracy": float((tp + tn) / (tp + tn + fp + fn)),
    }


def platt_scale(y_cal, p_cal, p_new):
    """Sigmoid (Platt) calibration fit on calibration predictions.

    Turns raw model probabilities into calibrated ones using a logistic
    regression on the logit of the raw score. Applies to any model family, so
    sklearn models and Keras models share the same honest-probability path.
    """
    from sklearn.linear_model import LogisticRegression

    p_cal = np.asarray(p_cal, float)
    p_new = np.asarray(p_new, float)

    def logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    reg = LogisticRegression(C=1e3)
    try:
        reg.fit(logit(p_cal).reshape(-1, 1), to_int(y_cal))
    except ValueError:
        return p_new.copy()
    return reg.predict_proba(logit(p_new).reshape(-1, 1))[:, 1]