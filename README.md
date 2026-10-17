# Motor_PM — Bearing Fault Detection from Motor Current (MCSA)

Predictive-maintenance benchmark on the [Paderborn bearing dataset](https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter)
(32 bearings, 2,557 usable 4-second recordings, 6203 ball bearings). The goal is a
**healthy vs damaged** detector driven **only by the motor phase currents** (MCSA),
since the target deployment has no vibration sensor. Vibration signals are kept as
a reference for exploration, never as model features.

## Setup

### 1. Prerequisites

- Python 3.11+ (developed on 3.14)
- `git`

### 2. Clone & create the environment

```bash
git clone https://github.com/MarwaneZe/Bearing-Faults-Predictive-Maintenance.git
cd Bearing-Faults-Predictive-Maintenance

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e .                   # core dependencies
pip install -e ".[notebook]"       # to run the notebooks
```

### 3. Get the dataset (not included in the repo — ~21 GB)

The raw data is **not** committed (see `.gitignore`). Download the Paderborn
bearing dataset from
<https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter>
and place the `.mat` files under `data/raw/<bearing_id>/`, e.g.:

```
data/raw/K001/N15_M07_F10_K001_1.mat
data/raw/KA04/...
```

32 bearing folders: `K001–K006` healthy, `KA..` artificial damage, `KI..` real
damage, `KB..` combination damage.

### 4. Build & run

```bash
jupyter lab     # then pick the .venv kernel you just created
```

Run the notebooks in order: `01_exploration` → `02_cleaning` →
`03_feature_engineering` → `04_models` → `05_detector` → `06_transfer`.
Notebooks 05/06 train Keras models with the JAX backend (selected automatically;
no TensorFlow required). Notebooks 02–04 write the processed CSVs into
`data/processed/` on first run.

## Data decisions (locked during exploration)

- **Option B — no trimming.** Recording length varies (~250k–299k samples at 64 kHz).
  Signals are kept whole; fixed sliding windows (16384 samples = 256 ms, 50 % overlap)
  absorb the variation. Trailing partial windows are dropped.
- **Corpus:** 2,557 recordings — 6 healthy bearings (480 recs), 12 artificial damage
  (KA, 957), 11 real damage by accelerated lifetime (KI, 880), 3 combination (KB, 240).
  Excluded: one corrupt (`KA08/N15_M01_F10_KA08_2.mat`) and two short signals
  (`KA08/N15_M07_F10_KA08_11.mat`, `KA05/N15_M07_F10_KA05_16.mat`).
- **Evaluation unit is the bearing**, never the recording: 80 recordings of one
  bearing are near-repeats, so recording-level metrics are inflated.
- **Splits are bearing-wise and assigned before windowing** — no cross-partition leak:
  - *Split A:* stratified 5-fold grouped CV for model building.
  - *Split B:* deployment verdict, run **once** — train on artificial + healthy,
    test on real/combination + healthy (the six healthy bearings are the shared
    reference, the standard protocol for this dataset).
- Metrics: **AUROC / PR-AUC** (majority ≈ 70 %); confusion at threshold 0.5.

## Pipeline

```
src/utils/     exploration helpers (loaders, parsers, metadata, audit, plausibility,
               bearings taxonomy, features, spectral, splits, metrics)
src/models/    six-model zoo (logreg, xgboost, random_forest, svm_rbf)
src/deep/      Keras (JAX backend) MLP + 1D-CNN + raw-window cache builder
src/experiments.py  shared orchestration for notebooks 05/06
src/pipeline/  production pipeline
  config.py            window/channel/feature configuration
  window_index.py      02_cleaning   -> data/processed/window_index.csv
  features.py          03_feature_engineering -> features_window.csv
  model.py             04_models     -> data/models/{model.joblib,config.json}
  predict_current.py   deployment CLI + predict_stream() API
```

### Notebook guide

See **Setup → 4. Build & run** above to install and launch. Notebooks:
- `01_exploration.ipynb` — the decision trail (corpus → integrity → taxonomy →
  time-domain → frequency/MCSA → split design → key decisions).
- `02_cleaning.ipynb` — builds the split-tagged window index (**76,868 windows**).
- `03_feature_engineering.ipynb` — extracts the per-window feature table
  (36 features/row: time-domain baseline + MCSA sideband/energy ratios).
- `04_models.ipynb` — Split A model selection + Split B deployment verdict
  (baseline: LR vs gradient boosting).
- `05_detector.ipynb` — **Experiment 1**: six independent models
  (`logreg`, `xgboost`, `random_forest`, `svm_rbf`, `mlp`, `1d-cnn`) compared by
  bearing-wise 5-fold CV over 26 training bearings (2 seeds), with a 6-bearing
  hold-out (2 healthy + 2 artificial + 2 real) used only for the final demo.
  Selects a winner by mean per-condition bearing-AUROC, calibrates it on
  out-of-fold predictions, sets a ≤10 %-healthy-false-alarm threshold, then
  runs the finished pipeline on the never-seen files. Executes in ~35–45 min
  (the CNN dominates the runtime).
- `06_transfer.ipynb` — **Experiment 2**: the same six models trained on
  artificial+healthy only, trialled ONCE on real/combination + healthy — the
  honest artificial→real generalization number, plus a real-only (KI) subtable.

### Deployment

```bash
# single recording  (Paderborn .mat or two npy current arrays)
python -m src.pipeline.predict_current --mat data/raw/KI04/N15_M07_F10_KI04_1.mat --speed 1500
python -m src.pipeline.predict_current --npz cur1.npy cur2.npy --speed 1500
```

```python
from pipeline.predict_current import load_model, predict_stream
model, cfg = load_model()                              # data/models/model.joblib
df = predict_stream(phase1, phase2, speed_rpm=1500, model=model, cfg=cfg)  # per-window p_damaged
```

The verdict is `damaged` if mean `p_damaged >= 0.5`. 256 ms windows, 50 % overlap.

## Results (Split B — artificial → real transfer, run once)

| Metric | Value |
|---|---|
| Window AUROC / PR-AUC | 0.78 / 0.85 (baseline 0.70) |
| Bearing AUROC per condition | 0.77–0.89 (max at low-load `N15_M01_F10`) |

Model chosen: logistic regression on scaled/imputed window features (Split A
bearing AUROC 0.80 vs 0.64 for gradient boosting).

**Honest read:** MCSA current features separate healthy from real damage clearly
above chance, but the artificial→real transfer is **moderate, not perfect** —
at threshold 0.5, 8 of 24 healthy bearing-condition pairs in the test set are
false-alarmed (see `data/processed/results/split_b_bearings.csv`). Field use should
budget further validation and a threshold tuned to the target false-alarm budget.

## Experiments 1 & 2 (six-model comparison, notebooks 05/06)

Extension that swaps the two-model baseline for **six independent models** (one
per ML family, no ensembles: logistic regression, XGBoost, random forest, an
RBF-kernel approximation, a feature MLP and a 1D-CNN on raw current windows).
Every result is printed as a comparison table; only the best model is carried to a
calibrated decision threshold and presented.

Two deliberate deviations from a plain "spend more compute" setup:
- **SVM-RBF is an approximation:** `SVC(probability=True)` embeds a 5-fold CV
  intractable at ~60k windows, so `svm_rbf` uses random Fourier features
  (RBFSampler 512) + logistic regression.
- **Imbalance handling is asymmetric on purpose:** feature models use equal
  class weights (weighted variants kept window AUROC but inverted the
  bearing-level ranking); the nets use class weights because plain
  binary cross-entropy on a 15:85 healthy:damaged mix collapses them to a
  constant majority output. Both decisions are documented in code.

Run order / presentation path:
```bash
cd notebooks && jupyter notebook  # kernel "motors" (venv Python 3.14)
```
- open `05_detector.ipynb` ⇒ Run All (~35-45 min) ⇒ winner table + demo on the
  6 held-out bearings at the end.
- open `06_transfer.ipynb` ⇒ Run All (~12-15 min) ⇒ artificial→real verdicts.

Artifacts land in `data/models/exp1_<model>.*` + `data/processed/results/exp1_*.csv`.

## Layout

```
data/raw/       32 bearing folders (.mat)
data/processed/ metadata, window_index, features_window, results/
data/models/    model.joblib + config.json
notebooks/      01_exploration . 02_cleaning . 03_feature_engineering . 04_models
src/utils/      exploration helpers
src/pipeline/   production pipeline
docs/           milestones & roadmap
```