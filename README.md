# Spatio-Temporal Traffic Accident Hotspot Prediction

Predict which 500 m grid cells of Great Britain will have **at least two recorded road collisions in the
next month**, using UK road-safety collisions 2021-2025, spatial clustering (DBSCAN) and machine learning.

This README describes only what is implemented. Every number below is copied from `results/metrics.json`
(the dashboard reads the same file). Numbers from the pre-audit version of the project were produced by a
protocol that tuned on the test year; they are kept in `results/old_run/` and are **not** comparable.

## Key findings

1. **A cell's own collision history carries almost all of the signal.** Ranking cells by their collisions in the last 12 months alone reaches
   AP 0.220 on 2025, against 0.255 for the model. In permutation importance on 2024 validation, one history feature
   (`historical_avg_collisions`, the long-run mean monthly count) accounts for nearly all of the drop in AP (0.2075; the next feature is 0.0028).
   The short recent-lag features individually matter little.
2. **HistGradientBoosting beats the trailing-12-month baseline on average precision: +0.035 (95% grid-bootstrap CI +0.031 to +0.038, excludes zero).**
   Top-5% precision also improves (+0.014, CI +0.010 to +0.019). **Top-1% precision is not distinguishable from the baseline** (+0.016, CI -0.0002 to +0.030).
3. **Random forest, HistGradientBoosting and logistic regression are practically tied** on 2024 validation (AP 0.2512 / 0.2499 / 0.2492). RF vs HistGradientBoosting: CI includes zero.
   RF vs logistic regression: +0.0020, CI only just excludes zero for AP (+0.0003 to +0.0038) and includes zero for top-1% and top-5% precision.
   The rule fixed before tuning therefore picked HistGradientBoosting; a simpler logistic regression would have done about as well.
4. **DBSCAN cluster features gave no gain** (validation AP -0.0030, worse in 5 of 5 seeds), so they are not in the final model.
5. **Recall is capped at 91.7%** because only cells with >= 3 collisions in 2021-2023 are modelled (the active-grid filter, which avoids using 2024-2025 information).
   System-wide recall at the frozen threshold is 0.330.

## Headline result (2025 test, evaluated once with a frozen configuration)

| | Frozen model (HistGradientBoosting) | Trailing-12-month count | Persistence (last month) |
|---|---|---|---|
| Average precision (PR-AUC) | **0.255** (10.9x the base rate) | 0.220 | 0.075 |
| Precision in each month's top 1% of cells | 0.437 (18.6x base rate) | 0.422 | 0.229 |
| Recall in top 1% / 5% / 10% | 0.187 / 0.467 / 0.613 | 0.180 / 0.436 / 0.573 | 0.098 / 0.227 / 0.336 |
| ROC-AUC (reference only) | 0.862 | 0.837 | 0.656 |

Test set: 364,656 cell-months, 8,549 hotspots, **base rate 2.34%**. The model beats the trailing-12-month
baseline by +0.035 average precision (grid-bootstrap 95% CI +0.031 to +0.038), and by +0.014 precision in the
top 5% (CI +0.010 to +0.019). **The gain in top-1% precision (+0.016) is not distinguishable from zero**
(CI -0.0002 to +0.030). These test-set bootstrap CIs were computed **after** the 2025 run, from the saved predictions
(`src/test_bootstrap.py`); they are reporting only and change no point metric. Put plainly: a well-chosen machine-learning model is better than simply ranking cells by
their last-12-month collision count, but only modestly.

At the frozen decision threshold (raw score >= 0.629, chosen on 2024): precision 0.289, recall 0.360, F1 0.320,
flagging 2.9% of cell-months. Confusion matrix (tn / fp / fn / tp): 348,525 / 7,582 / 5,473 / 3,076.
Calibrated probabilities have Brier score 0.0198 versus 0.0229 for always forecasting the base rate.
Per-month average precision ranges from 0.198 (Feb) to 0.293 (Jun); see `results/metrics.json`.

**Coverage ceiling.** Only cells with >= 3 collisions in 2021-2023 are modelled. They contain 91.7% of all 2025
hotspot cell-months (8,549 of 9,320). System-wide recall is therefore capped at 91.7%: 0.330 at the frozen
threshold, and 0.171 / 0.428 / 0.563 in the top 1% / 5% / 10%.

## Problem and label

For a grid cell `g` and month `t`, using only collisions in months <= t, predict whether month `t+1` has
>= 2 collisions in `g` (`HOTSPOT_MIN_COLLISIONS = 2`). The label is a count rule, not a measure of danger.
The base rate is about 2.3% of cell-months, so precision-recall metrics are reported, not accuracy.

## Data and provenance

- **Source:** UK Department for Transport road-safety collision data (STATS19-format collision records), as stated by the project owner.
  The repository records **no URL, licence text or download date**; check the source's licence terms before redistributing.
- **File used:** a single CSV, `data/raw/collisions_raw.csv` (44 columns, 513,801 rows; collisions per `collision_year`:
  2021: 101,087; 2022: 106,004; 2023: 104,258; 2024: 100,927; 2025: 101,525). Cleaning (`src/preprocessing.py`) keeps 513,748 rows after
  its steps (duplicate removal, coordinate validation, date parsing), and keeps the 18 columns listed in `src/config.py` plus a parsed `datetime`.
- **Download date:** <fill in>
- **Not in git:** all of `data/` (raw, interim, processed), old pickled models, run logs and `results/val_scores.parquet` are git-ignored.
  See "Reproducibility" for what is versioned.

## Architecture

```
raw CSV -> preprocessing -> collisions_clean.csv
        -> spatial_grid   (EPSG:27700, 500 m squares)      -> collisions_with_grid.csv, grid_cells.csv
        -> hotspot_detection (DBSCAN on 2021-2023 only)    -> grid_cluster_features.csv   [ablation only]
        -> temporal_dataset (active grids, grid x month)   -> grid_monthly_collisions.csv
        -> feature_engineering (features <= t, label t+1)  -> ml_features.parquet
        -> train_validate  (tune on 2024, select, calibrate, threshold) -> models/, results/metrics.json
        -> validation_comparison (cluster ablation, bootstrap CIs)
        -> freeze_config   (tests must pass; hashes)       -> results/frozen_config.json
        -> evaluate_test   (once; lock file)               -> metrics.json[test], outputs/*
        -> app.py          (Streamlit; reads metrics.json + outputs)
```

| Stage | File |
|---|---|
| Cleaning | `src/preprocessing.py` (also `src/eda.py` for exploratory figures) |
| Grid | `src/spatial_grid.py` |
| DBSCAN | `src/hotspot_detection.py` |
| Monthly table | `src/temporal_dataset.py` |
| Features / split | `src/features.py`, `src/feature_engineering.py`, `src/splits.py` |
| Metrics | `src/evaluation.py` |
| Training / selection | `src/train_validate.py`, `src/validation_comparison.py` |
| Freeze / test | `src/freeze_config.py`, `src/evaluate_test.py`, `src/test_bootstrap.py` |
| Tests | `tests/` |
| Old pipeline | `legacy/` (not used), `results/old_run/` (old outputs) |

## Method details

**Grid.** Longitude/latitude are projected to EPSG:27700 and floored to 500 m cells (true squares; the original
degree-based grid produced cells about 0.5 km by 0.29 km). Cell centres, not collision means, are the coordinates.
121,680 cells contain at least one collision.

**Active cells.** A cell is modelled if it has >= 3 collisions in 2021-2023 (30,388 cells), which uses no
information from 2024 or 2025. Cells that first become active later are not modelled (see coverage ceiling).

**Features (17)**, all computed from months <= t (`src/features.py`): cell centre latitude/longitude; month sin/cos;
collision counts at t, t-1, t-2; casualties at t; 3-, 6- and 12-month window sums; cumulative total, mean, maximum,
collision frequency and hotspot frequency; months since the last collision (a large sentinel if never seen).
`year` was removed (out of range at test time), as were duplicates that were rescalings of kept features.
The synthetic-grid unit tests prove that altering any count after month t leaves month t's features unchanged.

**Evaluation protocol.** Splits are by *target* month (the month being predicted):

| Split | Target months | Use |
|---|---|---|
| Train | <= 2023-12 | fit all models |
| Validation | 2024 | hyper-parameters, model selection, calibration, decision threshold |
| Test | 2025 | evaluated exactly once with `results/frozen_config.json` |

Training keeps all positives and samples negatives 1:4 (seed 42). **Trade-off:** the final model is trained only on
targets up to 2023-12, and 2024 is used for calibration and the threshold, so the most recent full year is not in the
model's fit. Refitting on 2021-2024 with a fixed threshold rule is listed under future work.

**Model selection (rule saved before any tuning, in `frozen_config.json`).** Tune logistic regression, random forest
(grid widened beyond the previous edge) and HistGradientBoosting on validation AP. Take the best by validation AP; if the top two are
within bootstrap noise (95% grid-bootstrap CI of the AP difference includes 0), take the simpler one
(order: logistic regression < HistGradientBoosting < random forest). Outcome on 2024 validation:

| Model | Validation AP |
|---|---|
| Random forest (depth 8, leaf 100) | 0.2512 |
| HistGradientBoosting (lr 0.03, 8 leaves, min leaf 50) | 0.2499 |
| Logistic regression (C = 0.01) | 0.2492 |
| Trailing-12-month baseline | 0.2158 |

RF vs HistGradientBoosting: +0.0013, CI [-0.0005, +0.0029], within noise, so HistGradientBoosting was chosen by the rule.
All three learned models are statistically close to each other (RF vs logistic regression +0.0020, CI [+0.0003, +0.0038]),
so the flexibility of tree ensembles is not clearly needed here. All three beat the trailing-12 baseline by about 0.033-0.035 AP
with clear CIs. The HistGradientBoosting and logistic-regression winners sit on the edge of their (unwidened) grids.

**Calibration and tiers.** The raw score is not a probability (1:4 sampling). A Platt calibrator (chosen over isotonic on the
second half of 2024, fitted on the first half) is refitted on all of 2024. On the second half of 2024 the Brier score was 0.0452 raw,
0.0204 calibrated, 0.0236 for a constant base-rate forecast. Reliability plot: `results/figures/reliability_val.png`
(calibrated curve is in-sample). **Risk tiers are rank-based within each month**: Critical = top 1% of cells, High = next 4%,
Medium = next 5%, otherwise Low. They replace the old fixed 0.5 / 0.7 / 0.9 probability bands.

## DBSCAN: role and negative result

DBSCAN is used to **define and analyse hotspot structure**, and cluster features were tested as predictors. **They did not
improve prediction and are not used in the final model.**

- DBSCAN is fitted on 2021-2023 collisions only (haversine, lat/lon in radians). Parameters were chosen from training data by a
  documented criterion: largest cluster <= 5% of clustered points (no chaining), then maximise clustered collisions
  (`results/dbscan_sweep.csv`). The original setting (0.3 km, min 10) chains: on 2021-2023 data its largest cluster holds 34.9% of clustered points
  (in the original 5-year fit, one cluster of 100,822 collisions covered Greater London).
- Chosen: eps = 0.1 km, min_samples = 10: 2,161 clusters, median size 14, largest cluster 2.2% of clustered points.
  These are tight clusters and cover only **16.9% of training collisions**, unlike the original broad zones (63% of collisions).
- Features: `grid_in_cluster` (>= 50% of the cell's training collisions are clustered) and `cluster_collision_density`.
- Ablation on 2024 validation, same model and hyper-parameters, 5 seeds: AP 0.2510 without, 0.2480 with (-0.0030; worse in 5 of 5 seeds;
  single-seed grid-bootstrap CI for the difference -0.0049 to -0.0021).
- Untested hypothesis for why they do not help: for training rows the cluster features are built from the same 2021-2023 years that produce the labels,
  so they may look more informative in training than they can be on later data. This was not tested.

## Dashboard

`streamlit run app.py` - pages: Home, Hotspot Predictions (Jan 2026 forecast, filters, CSV download), Risk Map (forecast or 2025 retrospective,
tier colours, actual hotspots as rings), Model Performance (test vs baselines, recall on modelled cells and system-wide, confusion matrix, calibration,
per-month AP, validation, bootstrap CIs, ablations), Feature Importance (permutation importance on validation).
All numbers come from `results/metrics.json` or saved outputs; `tests/test_dashboard.py` runs every page and checks displayed numbers against `metrics.json`.
The dashboard says "score" and "risk tier", and "estimated probability" only for calibrated values. The Jan 2026 forecast is produced with
the frozen model from data through Dec 2025 and has no outcome data, so it is unvalidated.

## Reproducibility

**Environment used:** Windows 11, Python 3.13.1, pandas 3.0.5, numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.0, pyarrow 25.0.1, pyproj 3.8.0,
joblib 1.6.0, matplotlib 3.11.1, seaborn 0.13.2, streamlit 1.63.0, pydeck 0.9.3, pytest 9.1.1 (pinned in `requirements.txt`).
Random seeds are fixed (`RANDOM_STATE = 42`; multi-seed ablations use seeds 42, 0, 1, 2, 3). Bit-identical results on other platforms or library versions are not guaranteed.

**Commands from a clean clone:**

```
git clone <repo-url> && cd <repo>
git checkout final-audited
python -m venv venv
venv\Scripts\activate                      # Windows; on Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
# place the raw file (not in git) at data/raw/collisions_raw.csv
python -m pytest tests -q                   # leakage, split, tier, selection-rule, dashboard tests
streamlit run app.py                        # reads results/metrics.json and outputs/ (both are in git)
```

To regenerate the intermediate data and validation results from the raw file (`python run_pipeline.py`, or its stages one by one):
preprocessing, grid, monthly table, DBSCAN, features, `train_validate.py`, `validation_comparison.py`.
**In this repository the 2025 results are already recorded**: `results/test_evaluated.lock` exists, so `train_validate.py`,
`validation_comparison.py` and `evaluate_test.py` all refuse to run and cannot overwrite the frozen model, `metrics.json` or `frozen_config.json`.
Re-running the whole pipeline is therefore only possible in a copy without the lock file, and would define a new protocol (a new test evaluation);
the original 2025 evaluation must not be repeated.

**Git markers:** branch `final-audited` is the audited final state. Tag `v1-2025-test-evaluated` marks the commit that first contains
`frozen_config.json`, the 2025 test results and the lock file (its dashboard and README are older than the branch tip).

**What is versioned:** all source, tests, `results/` (including `frozen_config.json`, `metrics.json`, `test_evaluated.lock`, figures, tuning tables, `old_run/` outputs),
`outputs/` (Jan 2026 forecast and 2025 test predictions) and the two small frozen artefacts `models/final_model.joblib` and `models/calibrator.joblib`
(their SHA-256 hashes are in `frozen_config.json`). **Not versioned:** everything under `data/`, superseded pickled models
(kept locally in `results/old_run/models/`), logs, `results/val_scores.parquet`.

**Verification status:** the pipeline stages were run individually and the test suite and dashboard were run; `run_pipeline.py` itself and a
from-scratch rebuild on a clean clone have not been executed end to end.

## Limitations

- **Spatial autocorrelation.** Neighbouring cells are correlated; metrics treat cells as independent, so the true uncertainty is larger than the CIs suggest.
  No spatial cross-validation was done.
- **One test year.** 2025 is a single draw. Bootstrap CIs resample grids, not years, and ignore model-seed variation.
- **Coverage ceiling.** 91.7% of 2025 hotspot cell-months are in modelled cells; grids first active after 2023 are invisible to the model.
- **Model differences are small.** The three model families are within noise of each other and only modestly better than the trailing-12-month baseline;
  top-1% precision does not separate from the baseline on the test year.
- **Tuning limits.** HistGradientBoosting and logistic-regression winners were on the edge of their grids and were not re-tuned.
- **Training window.** Trained on targets to 2023-12 only; 2024 data are used for calibration/threshold, not in the model fit.
- **Calibration** was fitted on one year; the reliability plot for the calibrated scores is in-sample. Calibration may drift.
- **Label and data.** The label is a count rule on police-recorded injury collisions; no traffic-volume, road-network or weather features are used,
  so it predicts where recorded collisions cluster, not causes. Cell boundaries (modifiable areal unit) affect results.
- **DBSCAN** produced no predictive gain; its parameters were chosen by a chaining criterion, not by predictive performance.
- **Old results** (AUC 0.8508, F1 0.2621) came from a test-tuned protocol with different grids and features and must not be compared with the numbers above.

## Disclosures about the 2025 evaluation

- The first attempt at `evaluate_test.py` crashed while unpickling the calibrator (its class was recorded as `__main__.PlattCalibrator`), before any 2025
  data were scored, and no lock or results were written. Only the loader in `evaluate_test.py` was fixed (a regression test was added); the frozen artefacts and hashes were untouched.
- Test-set bootstrap CIs (`src/test_bootstrap.py`) were added after the evaluation as reporting on the saved predictions; they change no point metric.

## Future work

- Refit on 2021-2024 with a fixed threshold/tier rule and re-evaluate on a later year.
- Spatially blocked cross-validation; models that use neighbouring-cell information.
- Handle cells that first become active after the training period; add exposure, road-network and weather covariates.
- Test the hypothesis about in-sample cluster features (for example, clusters fitted on strictly earlier years than the label).
- Widen the HistGradientBoosting and logistic-regression tuning grids.
