# Milestones & roadmap

The project was built as exploration Blocks (0–11) that map onto implementation
milestones. Phase 2 (pipeline + model) is done; these are the tracked units for
review, reproduction, and what comes next.

## Completed

- **M0 — Corpus inventory & integrity** (Blocks 0–5)
  Metadata builder, filename parser, channel catalog, audit grid, plausibility
  scanner. Excluded 3 files (1 corrupt, 2 short). Corpus = 2,557 recordings.
- **M1 — Damage taxonomy** (Block 6)
  `build_bearings_table`: healthy (6), artificial (12), real (11), combination (3).
- **M2 — Exploration & decisions** (Blocks 7–10)
  Time-domain features rejected at bearing level; frequency/MCSA direction;
  binary problem; bearing-wise Split A + Split B; Option B (no trimming).
  Written into `01_exploration.ipynb > Key decisions`.
- **M3 — Cleaning: split-tagged window index** (`02_cleaning`)
  76,868 windows; splits assigned per recording **before** windowing; leak verified.
- **M4 — Feature engineering** (`03_feature_engineering`)
  36 features/window: time-domain baseline + MCSA sideband/energy ratios per phase.
- **M5 — Modelling & deployment verdict** (`04_models`)
  Logistic regression chosen via Split A (bearing AUROC 0.80). Split B (artificial→
  real, run once): window AUROC 0.78, bearing AUROC 0.77–0.89 per condition.
  Artifacts: `data/models/*`, `data/processed/results/*`.
- **M6 — Deployment predictor** (`predict_current.py`)
  CLI + `predict_stream()` API; threshold 0.5 on mean window probability.
- **M7 — Six-model comparison** (`05_detector`, `06_transfer`)
  - Exp 1: 26-bearing CV across `logreg / xgboost / random_forest / svm_rbf /
    mlp / cnn` (mean per-condition bearing-AUROC over 2 split seeds; outputs in
    `data/processed/results/exp1_cv_table.csv`), 6-bearing hold-out demo with a
    calibrated ≤10 %-false-alarm threshold.
  - Exp 2: artificial→real transfer of all six + KI-only subtable.
  - Deliberate calls documented in-code: `svm_rbf` = Random-Fourier RBF
    approximation; feature models use equal class weights (weighted variants
    inverted bearing-level AUROC); nets use class weights (plain BCE collapses
    them to the majority prior); OOF predictions are row-aligned before any
    bearing-level aggregation (a mislocation bug inverted these numbers during
    development and is fixed).

## Next (proposals)

- **M7 — Robustness & thresholding.** Per-bearing dashboard; pick the decision
  threshold from the PR curve for a target false-alarm budget; evaluate Split B
  confidence intervals (bootstrap over bearings).
- **M8 — On-stream validation.** Sliding-window inference over continuous current;
  study window aggregation (mean vs majority vs EWMA) and latency (256 ms window).
- **M9 — Hardening.** Retrain pipeline as a CLI/package entry (`pipeline run`),
  pin environment (`requirements-lock.txt`), add run logging (MLflow), unit tests
  for `window_index`/`features` invariants.
- **M10 — Exploration follow-ups.** Envelope/cepstral current features; per-damage
  subtype reporting; data-insufficiency analysis (how many healthy/long recordings
  are needed to certify the 0.9-PR operating point).

## Validation rule

Every dataset/meta-number is reproducible from the notebooks; the split rule is
enforced by `verify()` and asserted at cleaning time. Never tune on Split B.