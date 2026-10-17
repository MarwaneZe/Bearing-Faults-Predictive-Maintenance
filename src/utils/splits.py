"""Evaluation split design.

Splits are BEARING-wise (recording-level splits would leak; recordings of one
bearing are near-repeats). A held-out bearing's ALL operating conditions stay
together, because condition is a confounder.

- Split A: pooled grouped CV over all 32 bearings (model building / tuning).
- Split B: train = artificial (KA) + healthy; test = real/combination (KI/KB)
           + healthy -- the deployment verdict, never tuned on.

Windowing happens AFTER these splits, never across partitions.
"""

import numpy as np

from .parsers import condition_key

FOLDS = 5  # recommended: >= 1 healthy + several damaged per fold


def _with_condition(metadata):
    df = metadata.copy()
    df["condition"] = df.apply(condition_key, axis=1)
    return df


def assign_folds(metadata, n_folds=FOLDS, seed=42):
    """Assign each bearing to exactly one CV fold (Split A).

    Stratified by label so EVERY fold keeps at least one healthy bearing.
    All recordings of a bearing (all 4 conditions) land in the same fold.
    Returns metadata + 'fold' column.
    """
    df = _with_condition(metadata)
    rng = np.random.default_rng(seed)
    labels = sorted(df["label"].unique())
    fold_of = {}
    for label in labels:
        bearings = sorted(df[df["label"] == label]["bearing"].unique())
        rng.shuffle(bearings)
        for i, b in enumerate(bearings):
            fold_of[b] = i % n_folds
    df["fold"] = df["bearing"].map(fold_of)
    return df


def assign_folds_on(window_df, n_folds=FOLDS, seed=42):
    """Bearing-wise folds over a WINDOW-level frame, stratified, seed-aware.

    Used by the model-comparison experiment: `window_df` carries every window
    of the training bearings; bearings are shuffled per label and bucketed
    round-robin into `n_folds`, so every fold keeps a healthy-damaged mix and
    no bearing (nor condition) straddles a fold.
    """
    rng = np.random.default_rng(seed)
    bdf = window_df.drop_duplicates("bearing")[["bearing", "label"]]
    fold_of = {}
    for label in sorted(bdf["label"].unique()):
        bearings = sorted(bdf.loc[bdf["label"] == label, "bearing"].tolist())
        rng.shuffle(bearings)
        for i, b in enumerate(bearings):
            fold_of[b] = i % n_folds
    return window_df.assign(fold=window_df["bearing"].map(fold_of))


def assignment_b(metadata):
    """Split B partition.

    Train = artificial (KA) + healthy; Test = real/combination (KI/KB) + healthy.
    Returns metadata + 'partition' column.
    """
    df = _with_condition(metadata)
    df["partition"] = np.where(
        df["bearing"].str.startswith(("KI", "KB")), "test", "train"
    )
    return df


def verify(metadata, split_df, split="A"):
    """Assert the leakage rules hold.

    - no bearing appears in more than one fold / partition
    - all 4 operating conditions of a bearing stay together
    """
    col = "fold" if split == "A" else "partition"
    per_bearing = split_df.groupby("bearing")[col].nunique().unique()
    assert (per_bearing == 1).all(), f"a bearing is split across {col}s!"
    for g, grp in split_df.groupby("bearing"):
        assert grp["condition"].nunique() == 4, f"{g}: operating conditions not together"
    print("verify OK:", split)

