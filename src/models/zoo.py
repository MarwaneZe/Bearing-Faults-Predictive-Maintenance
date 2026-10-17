"""Independent feature-based models for the comparison experiments.

Six models total, one per family, NO ensembles and no model depending on
another (per project rule).

All estimators are fit with EQUAL class weights. We initially used
imbalance-aware weights (class_weight='balanced' / scale_pos_weight): they
kept window-level AUROC (~0.73) but INVERTED the bearing-level ranking
(0.17 vs 0.80) on this small multi-condition corpus, so we standardised on
equal weights and document that as a negative-result finding.

Families:
- logreg       : linear baseline (current incumbent)
- xgboost      : gradient boosting (XGBoost)
- random_forest: bagged trees
- svm_rbf      : kernel method (explicit RBF-kernel approximation)
- mlp / cnn    : neural approaches live in src/deep/ (Keras, JAX backend)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.metrics import bearing_level, bearing_scores, to_int

FEATURE_MODELS = ("logreg", "xgboost", "random_forest", "svm_rbf")
ALL_MODELS = ("logreg", "xgboost", "random_forest", "svm_rbf", "mlp", "cnn")


def build_model(name, y, n_features=36):
    """Configure one estimator on equal footing (no class-weights)."""
    y = np.asarray(y).astype(int)
    if name == "logreg":
        return Pipeline([
            ("std", StandardScaler()),
            ("lr", LogisticRegression(max_iter=4000, C=1.0, random_state=0)),
        ])
    if name == "xgboost":
        import xgboost as xgb

        return xgb.XGBClassifier(
            n_estimators=600, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
            tree_method="hist", n_jobs=-1,
            eval_metric="auc", random_state=0,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=600, min_samples_leaf=4, max_features="sqrt",
            n_jobs=-1, random_state=0,
        )
    if name == "svm_rbf":
        return Pipeline([
            ("std", StandardScaler()),
            ("rbf", RBFSampler(n_components=512, gamma=1.0 / n_features, random_state=0)),
            ("lr", LogisticRegression(max_iter=4000, C=1.0, random_state=0)),
        ])
    raise ValueError(f"unknown feature model: {name}")


def fit_predict(model, X_train, y_train, X_test):
    """Fit and return P(damaged=1) on the test set."""
    y_train = to_int(y_train)
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


def _fold_row(test_frame, prob):
    y = to_int(test_frame["label"])
    bl = bearing_level(test_frame, prob)
    return {"window_auroc": _auroc(y, prob), "window_aupr": _aupr(y, prob), **bearing_scores(bl)}


def _auroc(y, p):
    from utils.metrics import auroc
    return auroc(y, p)


def _aupr(y, p):
    from utils.metrics import aupr
    return aupr(y, p)


def eval_cv(df, X, name, seed, n_folds=5, verbose=False):
    """Bearing-wise CV of one SKLEARN feature model over `df` windows.

    df: window frames with bearing/condition/label (aligned to X rows).
    X : (n_windows, 36) feature matrix aligning to df.
    seed shuffles the bearing buckets -> multi-seed ranking possible.
    Returns per-fold rows, out-of-fold (y, prob) for calibration, and the
    aggregate per-condition bearing scores.
    """
    from utils.splits import assign_folds_on

    df = df.reset_index(drop=True).copy()
    df = assign_folds_on(df, n_folds=n_folds, seed=seed)
    X = np.asarray(X)

    rows, oof_y, oof_p = [], None, None
    oof_p = np.full(len(df), np.nan)
    oof_y = np.zeros(len(df), dtype=int)
    for k in sorted(df["fold"].unique()):
        tr_mask, te_mask = df["fold"] != k, df["fold"] == k
        model = build_model(name, to_int(df.loc[tr_mask, "label"]), n_features=X.shape[1])
        prob = fit_predict(model, X[tr_mask.to_numpy()], df.loc[tr_mask, "label"], X[te_mask.to_numpy()])
        rows.append(_fold_row(df.loc[te_mask], prob))
        te_pos = np.where(te_mask.to_numpy())[0]
        oof_p[te_pos] = prob
        oof_y[te_pos] = to_int(df.loc[te_mask, "label"])

    oof = (oof_y, oof_p)
    bl = bearing_level(df, oof[1])
    return {
        "rows": pd.DataFrame(rows),
        "oof": oof,
        "bearing_scores": bearing_scores(bl),
    }