# Neighbor-Cell Feature Ablation — Results

Branch `neighbor-features`, off `main`. Full design rationale, ambiguity
flags, and the decision rule this ablation followed are in
[`neighbor_features_plan.md`](neighbor_features_plan.md); this document
records the actual results.

**Motivation.** Every existing feature in `src/features.py` and
`src/report_compliance/grid_month_aggregates.py` is per-grid-cell only.
README's own "Limitations" section already names this: *"Spatial
autocorrelation. Neighbouring cells are correlated; metrics treat cells as
independent... No spatial cross-validation was done."* This ablation tests
the more basic question first — before worrying about CI coverage under
spatial correlation, does giving the model neighbor-cell information as an
**input feature** actually improve prediction at all?

**What was added.** `src/neighbor_features.py`: for each grid cell and
month, the Moore-8 neighboring cells' `historical_avg_collisions` (the
existing leakage-safe cumulative rate), aggregated as `mean`, `max`, and
`neighbor_cell_count`. Computed over every occupied cell in the dataset, not
just the "active" (≥3 training-period collisions) subset that gets modelled
— a quiet neighbor is still real local signal. Leakage-tested the same way
as every other feature in this project
(`tests/test_neighbor_features_leakage.py`, 5 tests: future-independence,
current-month dependence, hand-computed values, edge-of-grid /
fewer-than-8-neighbors, and isolated-cell fallback).

**Nothing frozen was touched.** No code in this ablation reads or writes
`results/frozen_config.json`, `results/metrics.json`,
`results/test_evaluated.lock`, `models/final_model.joblib`,
`models/calibrator.joblib`, `data/processed/ml_features.parquet`,
`results/regression/regression_frozen_config.json`,
`results/regression/regression_test_evaluated.lock`,
`models/regression/final_model.joblib`, or
`results/regression/regression_metrics.json`. The 2025 test set was never
loaded or scored in either ablation.

---

## 1. Classifier ablation — null result

**Protocol.** Same frozen `HistGradientBoostingClassifier` config actually
in production (`learning_rate=0.03, max_leaf_nodes=8, min_samples_leaf=50`,
read read-only from `results/metrics.json`), 5 seeds, train ≤2023 / val
2024, `FEATURE_COLUMNS` (17 cols) vs `FEATURE_COLUMNS + NEIGHBOR_FEATURES`
(+3 cols) — same seed/bootstrap protocol as the existing DBSCAN
cluster-feature ablation (`src/validation_comparison.py::cluster_ablation`).

| metric | without neighbor | with neighbor | mean diff | seeds better |
|---|---|---|---|---|
| val AP | 0.2500 | 0.2510 | +0.0010 (sd 0.0010) | 4/5 |
| top-1% precision | 0.4351 | 0.4358 | +0.0007 | 2/5 |
| top-5% precision | 0.2213 | 0.2213 | −0.0000 | 2/5 |

**Grid-bootstrap 95% CI on the AP difference** (300 resamples, resampling
grids with replacement): point diff +0.0006, **CI [−0.0009, +0.0019]** —
includes 0.

**Verdict: `neighbor_features_help = False`.** Small and direction-consistent
(4 of 5 seeds positive) but not statistically distinguishable from noise at
this sample size. Per the pre-registered rule — keep only if the CI
excludes 0 on the positive side — this does not clear the bar. Full numbers:
`results/neighbor_features_ablation.json`.

## 2. Regression ablation — clear negative result

**Protocol.** Same frozen XGBoost config (`n_estimators=300, max_depth=4,
learning_rate=0.05`, read read-only from
`results/regression/regression_frozen_config.json`), 3 seeds (XGBoost here
has no negative-sampling step, so seed-to-seed variance is smaller — results
were in fact identical across all 3 seeds, expected given fixed data and no
subsampling), train ≤2023 / val 2024, `REGRESSION_FEATURE_COLUMNS` (24 cols)
vs `+ NEIGHBOR_FEATURES` (+3 cols).

| metric | without neighbor | with neighbor | diff | seeds better |
|---|---|---|---|---|
| val MAE | 0.2395 | 0.2433 | +0.0038 | 0/3 |
| val RMSE | 0.4309 | 0.4321 | +0.0012 | 0/3 |
| val R² | 0.1730 | 0.1682 | −0.0048 | 0/3 |

**Grid-bootstrap 95% CI on the MAE difference** (300 resamples): point diff
+0.0038, **CI [+0.0036, +0.0040]** — excludes 0, but on the *worsening*
side, consistently across all 3 seeds.

**Verdict: `neighbor_features_help = False`.** Unlike the classifier result,
this is not a null finding — it is a statistically clear **negative**
finding: adding these neighbor features measurably hurts the regression
model on every metric. Hypothesis: the regression target
(`next_month_collisions`) is extremely sparse (83.8% of grid-months have
zero collisions — see `docs/report_compliance_checklist.md` item 3); three
additional, comparatively noisy neighbor-derived columns likely gave
XGBoost extra ways to fit variance without adding compensating signal, on a
target where there is very little signal to find in the first place. Full
numbers: `results/regression/neighbor_features_ablation.json`.

## 3. Discipline followed

Both ablations used the same pre-registered decision rule as the existing
DBSCAN cluster-feature ablation
(`src/validation_comparison.py::cluster_ablation`,
`docs/report_compliance_checklist.md` item 8's underlying analysis): test on
validation only; keep the feature only if the 95% grid-bootstrap CI of the
validation-metric difference excludes 0 on the improving side. Neither
result cleared that bar, so — per the plan's pre-registered stopping rule —
**no one-shot 2025 test-set evaluation was spent on either track.** That
one-shot budget, for both the classifier and the regression track, remains
unused and available for a change that validation evidence says is clearly,
substantially better, not for an incremental feature that validation itself
says is noise or worse.

## 4. Overall takeaway

Three separate ablations now point the same direction:

1. **DBSCAN cluster features** (existing, pre-dating this branch) — no gain
   (validation AP 0.2480 with clusters vs 0.2510 without, per
   `README.md`'s Limitations section).
2. **Neighbor features, classifier track** (this branch) — null result, CI
   includes 0.
3. **Neighbor features, regression track** (this branch) — clear negative
   result, CI excludes 0 on the worsening side.

All three tried to add spatial context beyond each cell's own history, and
none improved on it. The consistent conclusion: at 500 m / monthly
resolution, on this data, the existing per-cell spatiotemporal feature set —
particularly `historical_avg_collisions` and the other cumulative
lag/rolling-count features — already captures most of the exploitable
signal. Naive additions of more spatial or cluster context (DBSCAN
membership, Moore-8 neighbor rates) do not improve on it, and can actively
hurt a model that has less signal to work with in the first place (the
regression track's sparse count target). This does not mean spatial
autocorrelation is unimportant — README's existing limitation about
uncertainty intervals ignoring it still stands, since that is a question
about *inference*, not about whether neighbor data helps *prediction* — but
it does mean the specific, concrete "give the model its neighbors' data"
fix tested here is not the way to improve these two models further.
