# DA1 Report-Compliance Checklist

This maps every claim in the original Digital Assignment 1 (DA1) proposal to
the exact file, output, or demo screen that proves it, on branch
`report-compliance`. It is meant to be usable directly in the viva: for each
claim, "where to click / what to open" plus the honest caveats.

**Two systems exist side by side.** The **frozen classifier** (binary
hotspot classification, `HistGradientBoosting`, rank-based 4-tier risk,
React+FastAPI dashboard) is the primary, rigorously audited, production
artifact — untouched by any of this work. The **report-compliance track**
(everything under `src/report_compliance/`, `tests/report_compliance/`,
`results/regression/`, `models/regression/`, `outputs/report_compliance/`)
is an additive, parallel demonstration built specifically to satisfy the
DA1 proposal's literal claims. See "Honest scope note" at the end for how
these two relate.

---

## 1. RF Regressor + XGBoost Regressor predicting collision COUNT (not classification)

| | |
|---|---|
| **Status** | ✅ Done, evaluated once on 2025 |
| **Evidence** | `src/report_compliance/train_validate_regression.py` (tuning), `src/report_compliance/retune_xgboost_widened.py` (edge-check re-tune before freezing), `models/regression/final_model.joblib` (frozen XGBoost), `results/regression/regression_frozen_config.json` (frozen hyperparameters + hashes), `results/regression/regression_comparison_table.csv` |
| **Target** | `next_month_collisions` (raw count per grid), NOT the binary `hotspot` label the classifier uses |
| **Chosen model** | XGBoost, `n_estimators=300, max_depth=4, learning_rate=0.05` — selected by the pre-registered rule ("top model clearly ahead of the runner-up" on validation MAE; bootstrap CI of the MAE gap vs Random Forest excluded 0) |
| **Caveat** | This is a genuinely separate model from the classifier; it is not used by, and does not feed into, the classifier's predictions or dashboard. |

## 2. Compared against a historical-average baseline

| | |
|---|---|
| **Status** | ✅ Done |
| **Evidence** | `src/report_compliance/regression_baseline.py` (per-grid, per-calendar-month mean, fit on train ≤2023 only, falls back grid-month → grid → global mean) |
| **Result** | Baseline included as a first-class candidate in every comparison table below, not just a footnote |

## 3. Evaluated with MAE / RMSE / R²

| | |
|---|---|
| **Status** | ✅ Done, both validation and test, all three candidates |
| **Evidence** | `src/report_compliance/regression_evaluation.py::evaluate_regression()` (all metrics computed on the RAW COUNT scale after `expm1()` inversion); full numbers in `results/regression/regression_metrics.json` and `results/regression/regression_comparison_table.csv` |

**Validation (2024 targets), n=364,656:**

| model | MAE | RMSE | R² |
|---|---|---|---|
| xgboost (chosen) | 0.2395 | 0.4309 | 0.1730 |
| random_forest | 0.2486 | 0.4372 | 0.1487 |
| historical_average_baseline | 0.2833 | 0.5115 | -0.1652 |

**Test (2025 targets, evaluated exactly once), n=364,656:**

| model | MAE | RMSE | R² |
|---|---|---|---|
| xgboost (frozen) | 0.2875 | 0.4660 | **0.0377** |
| historical_average_baseline | **0.2844** | 0.5126 | -0.1642 |

**Caveat — report honestly in the viva:** on test, XGBoost's MAE is
essentially tied with (very slightly *worse* than) the baseline's, even
though its RMSE and R² are clearly better. Validation had suggested a
larger, more one-sided win (MAE gap ~0.044 in XGBoost's favour) than
materialized on test (~0.003 in the *baseline's* favour). This is a
legitimate validation→test generalization gap on an extremely sparse
target (83.8% of grid-months have zero collisions at 500m resolution;
mean actual ≈0.176 collisions/grid-month) — not an error, and exactly the
kind of thing the "evaluate once, never re-tune after seeing test" rule
exists to surface honestly rather than hide.

## 4. Explicit hour / day-of-week / weekend / time-of-day features

| | |
|---|---|
| **Status** | ✅ Done, two layers |
| **Collision-level EDA** | `src/report_compliance/temporal_eda.py` → `outputs/figures/report_compliance/*.png` (4 charts: by hour, by day of week, weekend vs weekday, by time-of-day bucket) + `outputs/report_compliance/temporal_eda_*.csv` |
| **Grid-month features actually used by the regression model** | `src/report_compliance/grid_month_aggregates.py` — `historical_rush_hour_share`, `historical_weekend_share`, `historical_night_share` (cumulative, leakage-safe); these ARE in the frozen XGBoost model's feature list (see `results/regression/regression_frozen_config.json["features"]`) |
| **Leakage tests** | `tests/report_compliance/test_grid_month_aggregates_leakage.py` — same synthetic-grid method as the classifier's own `tests/test_features_leakage.py` (future-months-don't-leak-back, current-month-does-affect-its-own-feature, hand-computed values, never-seen sentinel) |
| **Caveat** | The FROZEN CLASSIFIER itself does not use any of these temporal features — it only ever used spatial + lag/rolling-count features and month sin/cos. Hour/day-of-week/weekend features exist only in the new regression track's feature set, by design (the classifier's feature set is frozen and untouched). |

## 5. Folium for interactive mapping

| | |
|---|---|
| **Status** | ✅ Done |
| **Evidence** | `src/report_compliance/build_folium_map.py` → `outputs/report_compliance/hotspot_map.html` (2.7MB static export, open directly in a browser) |
| **Content** | 1,849 markers (349 Critical / 1,000 High / 500 Medium, capped for legibility; Low tier excluded — same convention as the pre-existing `legacy/hotspot_map.py`). Each marker's popup shows the classifier's risk tier + probability, the new quantile tier, the new regression track's predicted count (XGBoost and baseline), and the actual 2025 count. |
| **Caveat** | Shown ALONGSIDE the React dashboard (`web/frontend`), which remains primary — this does not replace it. Markers show each grid's single "peak risk month" of 2025 (highest classifier probability), not all 12 months per grid, to keep the map browsable. |

## 6. GeoPandas / Shapely for spatial processing

| | |
|---|---|
| **Status** | ✅ Done, with a concrete equivalence proof |
| **Evidence** | `src/report_compliance/spatial_grid_geopandas.py` (GeoDataFrame reprojection via `to_crs`, Shapely `box` polygons per occupied cell, cell assignment via a genuine `geopandas.sjoin` spatial join — not a re-run of the same arithmetic) |
| **Equivalence test** | `tests/report_compliance/test_grid_geopandas_equivalence.py` — **exact `grid_id` match, 0 mismatches**, checked against a 20,000-row sample, the **full 513,748-row cleaned dataset**, AND hand-built synthetic points sitting exactly on cell boundaries |
| **Caveat** | The PRODUCTION grid assignment (`src/spatial_grid.py`) still uses `pyproj` + manual `numpy` floor-division, unchanged, by design — this GeoPandas/Shapely version is a parallel, proven-equivalent implementation built specifically to demonstrate the proposed tool, not a replacement. |

## 7. Filtering by road type and weather

| | |
|---|---|
| **Status** | ✅ Done in the Folium map; ❌ NOT added to the React dashboard |
| **Evidence** | `outputs/report_compliance/hotspot_map.html`'s `TagFilterButton` control, with tags `Road: Single carriageway` / `Road: Other road type` and `Weather: Adverse-weather prone` / `Weather: Predominantly fine weather`, derived from `historical_single_carriageway_share` / `historical_adverse_weather_share` (`src/report_compliance/grid_month_aggregates.py`) |
| **Column verification** | Confirmed against the actual cleaned data (not assumed): `road_type`, `weather_conditions`, `road_surface_conditions`, `speed_limit` all present in `data/interim/collisions_clean.csv`; value distributions cross-checked against the documented DfT STATS19 schema before choosing coarse, defensible binary splits (dominant category vs "other"), rather than betting correctness on every rare code |
| **Caveat** | The original plan said "Folium map **and/or** React dashboard" — only the Folium map got this filter. If the viva wants it in React too, that is unbuilt scope, not a broken claim. |

## 8. LOW/MEDIUM/HIGH risk tiers via quantile thresholds

| | |
|---|---|
| **Status** | ✅ Done, shown alongside the existing system |
| **Evidence** | `src/report_compliance/quantile_tiers.py` (per-month terciles, 33rd/66th percentile) → `results/regression/quantile_tiers_2025.parquet`; also surfaced in the Folium map's popups and filter tags |
| **Tests** | `tests/report_compliance/test_quantile_tiers.py` — tercile balance, per-month independence, monotonicity with score, and a byte-for-byte check that reading the frozen classifier's predictions never mutates that source file |
| **Caveat** | This is explicitly an ALTERNATE categorization next to the classifier's real, validated rank-based 4-tier system (Critical/High/Medium/Low, `src/evaluation.py::assign_tiers`) — it is not used for any operational decision, and it recomputes tiers from the classifier's own already-frozen scores, not a new model. |

---

## Cross-cutting compliance controls (not individual DA1 claims, but load-bearing for all of them)

| Control | Evidence |
|---|---|
| Regression track's own pre-registered selection rule, written and hashed BEFORE any tuning | `results/regression/regression_frozen_config.json` (`written_at_utc` in the registration step precedes `frozen_at_utc`) |
| Regression track's own frozen config + one-shot test lock, entirely separate from the classifier's | `results/regression/regression_frozen_config.json` / `results/regression/regression_test_evaluated.lock` — distinct paths from `results/frozen_config.json` / `results/test_evaluated.lock`, verified by hash at every step |
| Zero modification to any frozen classifier file, at every phase | `results/frozen_config.json`, `results/metrics.json`, `results/test_evaluated.lock`, `models/final_model.joblib`, `models/calibrator.joblib`, `data/processed/ml_features.parquet`, `data/processed/grid_monthly_collisions.csv` all confirmed byte-identical (md5) before and after every phase of this work |
| Full test suite | 58/58 passing (41 pre-existing classifier tests + 17 new report-compliance tests) |

---

## Honest scope note for the viva

**What the DA1 proposal asked for, framed as the core deliverable:** a
regression system (RF + XGBoost predicting count, vs a historical-average
baseline, MAE/RMSE/R²), with Folium mapping, GeoPandas/Shapely spatial
processing, explicit temporal features, road/weather filtering, and
quantile-based tiers, as the primary methodology.

**What was actually built and evaluated as the primary system (frozen,
pre-existing, untouched by this work):** a binary hotspot CLASSIFIER
(`HistGradientBoosting`, chosen over logistic regression and random forest
by a pre-registered rule), evaluated on 2025 with average precision 0.2549
and ROC-AUC 0.8624 against a 2.34% base rate, with rank-based
Critical/High/Medium/Low tiers and a React+FastAPI dashboard. This is
methodologically a stronger and better-validated system for the underlying
business question ("where should patrols/interventions be prioritized")
than a raw count regression would be, but it is a **differently-scoped**
answer than the one the DA1 proposal specified.

**What this report-compliance track adds:** a complete, honestly-evaluated,
separately-frozen regression track that satisfies every literal DA1 claim
(items 1-8 above), built entirely additively so the audited classifier is
never put at risk. Its own test-set result (item 3) is modest — XGBoost
essentially ties the baseline on MAE at test time, despite a clearer
validation-time win — which should be presented as a genuine, honestly
reported finding about how hard fine-grained count regression is on this
data, not glossed over.

**Recommended framing for the viva:** "the proposal's regression
methodology was fully implemented and evaluated with the same rigor as the
classifier (pre-registered selection rule, frozen config, one-shot test
evaluation) as an additive comparison track; the project's primary,
production-quality answer to the underlying problem is the classifier,
which we consider methodologically stronger for this task; the regression
track's more modest test performance is itself an honest and informative
result about the difficulty of the count-prediction framing at 500m grid
resolution."
