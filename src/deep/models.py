"""Keras (JAX backend) neural models: 1D-CNN on raw current + MLP on features.

Both are INDEPENDENT models (no ensemble) sharing the same honesty protocol as
the sklearn zoo: bearing-wise CV for ranking, out-of-fold probabilities for
calibration, physical class imbalance via class weights (no resampling).
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import pandas as pd
from keras import layers, models, regularizers
from keras.callbacks import EarlyStopping
from keras.utils import set_random_seed

from utils.metrics import bearing_level, bearing_scores, to_int
from utils.splits import assign_folds_on


def build_cnn(seed=0, input_len=1024, channels=2):
    """Small 1D-CNN for the raw (decimated, L2-normalised) current window."""
    set_random_seed(seed)
    inp = layers.Input((input_len, channels))
    x = layers.Conv1D(32, 32, strides=4, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv1D(64, 16, strides=4, padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    m = models.Model(inp, out)
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m


def build_mlp(seed=0, n_features=36):
    """Plain feed-forward net on the engineered features (the 'normal NN')."""
    set_random_seed(seed)
    inp = layers.Input((n_features,))
    x = layers.Dense(64, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(inp)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    m = models.Model(inp, out)
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m


def build_model(name, seed=0, n_features=36, input_len=1024, channels=2):
    if name == "mlp":
        return build_mlp(seed, n_features)
    if name == "cnn":
        return build_cnn(seed, input_len, channels)
    raise ValueError(f"unknown neural model: {name}")


def _class_weight(y):
    y = to_int(y)
    neg, pos = int((y == 0).sum()), max(int((y == 1).sum()), 1)
    return {0: 1.0, 1: neg / pos}


def fit_net(Xtr, ytr, Xval, yval, name, seed, epochs=40, batch_size=256, patience=4):
    """Train a Keras net with early stopping on a bearing-wise val set.

    Nets get class weights: without them, plain binary cross-entropy on the
    15:85 healthy:damaged mix collapses to a constant majority output (BCE is
    minimised by predicting the empirical class rate for every window).
    """
    m = build_model(name, seed=seed)
    ytr, yval = to_int(ytr).astype(np.int32), to_int(yval).astype(np.int32)
    es = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    m.fit(
        Xtr, ytr, validation_data=(Xval, yval),
        epochs=epochs, batch_size=batch_size, verbose=0,
        callbacks=[es], class_weight=_class_weight(ytr),
    )
    return m


def eval_cv_features(df, X, name, seed, n_folds=5):
    """Bearing-wise CV for the feature MLP."""
    df = df.reset_index(drop=True).copy()
    df = assign_folds_on(df, n_folds=n_folds, seed=seed)
    X = np.asarray(X)
    rows, oof_p, oof_y = [], np.full(len(df), np.nan), np.zeros(len(df), dtype=int)
    for k in sorted(df["fold"].unique()):
        tr, te = df["fold"] != k, df["fold"] == k
        m = fit_net(X[tr.to_numpy()], df.loc[tr, "label"],
                    X[te.to_numpy()], df.loc[te, "label"], name, seed)
        prob = m.predict(X[te.to_numpy()], verbose=0).ravel()
        rows.append(_net_fold_row(df.loc[te], prob))
        te_pos = np.where(te.to_numpy())[0]
        oof_p[te_pos] = prob
        oof_y[te_pos] = to_int(df.loc[te, "label"])
    oof = (oof_y, oof_p)
    return {"rows": pd.DataFrame(rows), "oof": oof,
            "bearing_scores": bearing_scores(bearing_level(df, oof[1]))}


def eval_cv_cache(df, cache, name, seed, n_folds=5, train_stride=2):
    """Bearing-wise CV for the 1D-CNN over the raw-window cache.

    train_stride subsamples every stride-th window per recording during
    TRAINING only (windows are 50%-overlapping near-repeats, so this cuts
    runtime with negligible learning cost); evaluation always uses all windows.
    """
    df = df.reset_index(drop=True).copy()
    df = assign_folds_on(df, n_folds=n_folds, seed=seed)
    pos = _positions(df)
    rows, oof_p, oof_y = [], np.full(len(df), np.nan), np.zeros(len(df), dtype=int)
    for k in sorted(df["fold"].unique()):
        tr, te = df["fold"] != k, df["fold"] == k
        tr_pos = np.where(tr.to_numpy())[0]
        if train_stride > 1:
            wids = df["window_id"].to_numpy()[tr_pos]
            tr_pos = tr_pos[wids % train_stride == 0]
        Xtr = np.asarray(cache[pos[tr_pos]], dtype=np.float32)
        ytr = df["label"].to_numpy()[tr_pos]
        te_pos = np.where(te.to_numpy())[0]
        Xte = np.asarray(cache[pos[te_pos]], dtype=np.float32)
        m = fit_net(Xtr, ytr, Xte, df["label"].to_numpy()[te_pos], name, seed)
        prob = m.predict(Xte, verbose=0).ravel()
        rows.append(_net_fold_row(df.loc[te], prob))
        oof_p[te_pos] = prob
        oof_y[te_pos] = to_int(df.loc[te, "label"])
    oof = (oof_y, oof_p)
    return {"rows": pd.DataFrame(rows), "oof": oof,
            "bearing_scores": bearing_scores(bearing_level(df, oof[1]))}


def _net_fold_row(test_frame, prob):
    y = to_int(test_frame["label"])
    from utils.metrics import auroc, aupr

    bl = bearing_level(test_frame, prob)
    return {"window_auroc": auroc(y, prob), "window_aupr": aupr(y, prob),
            **bearing_scores(bl)}


def _positions(df):
    """Cache row indices aligned to df via (recording, window_id)."""
    from pipeline.config import DEEP_KEYS

    keys = pd.read_csv(DEEP_KEYS)
    keys["pos"] = np.arange(len(keys))
    m = df[["recording", "window_id"]].merge(
        keys, on=["recording", "window_id"], how="left")
    if m["pos"].isna().any():
        raise ValueError("cache keys missing for some windows in df")
    return m["pos"].to_numpy(dtype=np.int64)