# Report-Compliance Track — Plan

## Purpose

The frozen pipeline on `main` (binary hotspot classifier, `HistGradientBoosting`, rank-based
4-tier risk, React+Streamlit dashboards) is methodologically sound but does not literally match
the Digital Assignment 1 (DA1) proposal. This plan adds a **parallel, additive
"report-compliance" track** that demonstrates every DA1 claim without touching the frozen
classifier, its config, metrics, lock file, model, or calibrator.

**Hard constraint — never modify:**
`results/frozen_config.json`, `results/metrics.json`, `results/test_evaluated.lock`,
`models/final_model.joblib`, `models/calibrator.joblib`, `data/processed/ml_features.parquet`,
or any code path that produces them (`src/freeze_config.py`, `src/evaluate_test.py`,
`src/train_validate.py`, `src/features.py:FEATURE_COLUMNS`, `src/evaluation.py:assign_tiers`
as currently wired into the frozen run).

All new work lives in new files/directories: `src/report_compliance/`, `data/processed/regression_*`,
`models/regression/`, `results/regression/`, `outputs/report_compliance/`, `docs/report_compliance_checklist.md`,
new tests under `tests/report_compliance/`.

## Current-state facts (from repo inspection, read-only)

- Grid: `src/spatial_grid.py` uses `pyproj` (WGS84→EPSG:27700) + manual `np.floor` binning into
  500 m cells (`GRID_SIZE_METERS`, `src/config.py`). No GeoPandas/Shapely anywhere in the repo.
- Features: `src/features.py:FEATURE_COLUMNS` (~lines 22-40) is spatial + lag/rolling-count +
  month sin/cos only. **No hour, day-of-week, weekend, or rush-hour features exist.**
  `day_of_week` and `time` exist as raw columns in `data/interim/collisions_clean.csv` but are
  dropped before feature engineering — this is exactly the additive gap to fill.
- Split: `src/splits.py::assign_split()` — train ≤ 2023-12, validate 2024, test 2025, split by
  **target month** (t+1). Constants in `src/config.py`.
- Lock: `src/freeze_config.py` writes `results/frozen_config.json` with sha256 hashes;
  `src/evaluate_test.py` refuses to re-run if `results/test_evaluated.lock` exists and verifies
  hashes before evaluating. **The new regression track needs its own lock file** — proposed
  `results/regression/test_evaluated.lock` + `results/regression/frozen_config.json`, produced by
  a new (copied/adapted, not shared) freeze/evaluate script so the frozen classifier's lock logic
  is never touched or re-triggered.
- Tiers: `src/evaluation.py::assign_tiers()` (~lines 132-149) — rank-based, widest→narrowest,
  tiers `Critical / High / Medium / Low` from `frozen_config.json["risk_tiers"]["fractions"]`.
- Dashboards: React (`web/frontend`) + FastAPI (`web/backend`) is primary; Streamlit is legacy
  (`legacy/streamlit_dashboard`). Folium exists only in `legacy/hotspot_map.py` (superseded) and
  as a static output `results/old_run/accident_hotspot_risk_map.html`. No Folium in `src/` or `web/`.
- Road/weather columns exist in `data/interim/collisions_clean.csv` (`road_type`, `speed_limit`,
  `junction_detail`, `junction_control`, `light_conditions`, `weather_conditions`,
  `road_surface_conditions`, `urban_or_rural_area`) but are **not** carried into `ml_features.parquet`.
- Leakage tests: `tests/test_features_leakage.py` is the existing standard (synthetic-grid,
  feature-at-month-t-uses-only-≤t, target-is-next-month, split-by-target-month). New temporal/
  aggregate features must get equivalent tests.
- Dependencies: root `requirements.txt` has no `geopandas`, `shapely`, `folium`, or `xgboost`.
  All four are net-new additions.

## Item-by-item plan

### 1. Regression track (count prediction)

- **Target:** next-month collision **count** per grid (reuse the existing pre-binarization count
  that presumably already exists in the monthly grid table before `hotspot_detection.py`
  thresholds it — confirm exact source column in `data/interim`/`data/processed` monthly CSV
  during implementation; do not re-derive from raw data if an existing count column is available).
- **Target transform — proposed: `log1p(count)`, fit and validate in log space, then
  `expm1()` back before computing MAE/RMSE/R²/reporting.** Justification: collision counts per
  500 m grid-month are heavily right-skewed (many zero/low-count cells, few high-count cells),
  which is exactly the same skew the existing pipeline handles via `historical_hotspot_frequency`-
  style features and log-ish month encodings; log1p stabilizes variance and prevents a few
  high-count grids from dominating squared-error loss, at the cost of the trained model's raw-scale
  errors being slightly biased low unless corrected. Report metrics **on the original count scale**
  (after inverse-transforming predictions) since that's the unit MAE/RMSE/R² are meaningfully
  interpreted in for the report — log-space metrics reported as a secondary diagnostic only, not
  the headline numbers.
- **Split:** identical scheme to the classifier (train ≤ 2023-12, val 2024, test 2025, split by
  target month) — reuse `src/splits.py::assign_split()` directly (read-only reuse, not a copy)
  since it's generic and doesn't touch frozen artifacts.
- **Models:** (a) historical-average baseline — mean count for that grid over the training window
  (and/or grid+month-of-year average, matching the "historical-average baseline" language in the
  proposal — confirm which averaging granularity with you before implementation), (b)
  `RandomForestRegressor`, (c) `XGBRegressor`.
- **Metrics:** MAE, RMSE, R² per model on val and on test (test evaluated once, behind the new
  regression-specific lock file), plus a comparison table (markdown + CSV) mirroring
  `results/metrics.json`'s style but under `results/regression/`.
- **New dependency:** `xgboost`.

### 2. Explicit temporal features

- **Collision-level EDA additions:** hour, day-of-week, weekend/weekday, time-of-day bucket
  (e.g. night/morning-rush/midday/evening-rush/evening) computed directly from
  `data/interim/collisions_clean.csv`'s existing `time`/`day_of_week` columns. Pure analysis/plots
  (new notebook or `src/report_compliance/temporal_eda.py`), no pipeline coupling.
- **Grid-month aggregate features (for the regression track):** rush-hour share, weekend share,
  night share, etc., aggregated per grid **per target month's training window** (i.e. computed
  from the same month(s) of data the existing lag/rolling features use — contemporaneous-month
  aggregates are already the pattern `collisions_lag_0` uses, so this is consistent with the
  existing leakage-safety standard).
- **Leakage testing:** add `tests/report_compliance/test_temporal_features_leakage.py` mirroring
  `tests/test_features_leakage.py`'s synthetic-grid method — assert aggregate features at month t
  use only that month's (or earlier months') raw collision rows, and that split-by-target-month
  still holds.

### 3. Folium map

- Static exported HTML (`outputs/report_compliance/hotspot_map.html`), built from **read-only**
  reads of `models/final_model.joblib`/`results/metrics.json` predictions (via a read-only loader,
  no re-inference through frozen lock logic) plus the new regression track's predicted counts.
  Layers: classifier risk tier, quantile tier (item 6), regression predicted count, with
  road-type/weather filters (item 5) as Folium layer controls / dropdowns.
- Presented **alongside**, not replacing, the React dashboard.
- **New dependency:** `folium`.

### 4. GeoPandas/Shapely grid rewrite

- New module `src/report_compliance/spatial_grid_geopandas.py`: build the same 500 m grid using
  `geopandas`/`shapely` (`shapely.geometry.box` polygons over the EPSG:27700 extent, spatial join
  via `gpd.sjoin`) instead of manual `pyproj` + `np.floor` binning.
- **Explicit equivalence test** (`tests/report_compliance/test_grid_geopandas_equivalence.py`):
  run both implementations over the same collision sample and assert identical `grid_id` assignment
  per row (or, if projection/rounding differs at cell boundaries, assert equivalence within a
  documented tolerance and report the mismatch rate).
- **New dependencies:** `geopandas`, `shapely` — **flagged as highest install risk on Windows**
  (see Risks below).

### 5. Road type / weather filters

- Join `road_type`, `weather_conditions` (and optionally `light_conditions`,
  `road_surface_conditions`) from `data/interim/collisions_clean.csv` into a grid-month aggregate
  (e.g. dominant/most-frequent category per grid-month, or category shares) — new columns in the
  new regression feature table, not in the frozen `ml_features.parquet`.
- Exposed as filters in (a) the Folium map (item 3) and (b) optionally the React dashboard, as a
  new read-only endpoint/panel in `web/backend` + `web/frontend` that does not alter existing
  dashboard routes/components.

### 6. Quantile-based LOW/MEDIUM/HIGH tiers

- New function (e.g. `src/report_compliance/quantile_tiers.py::assign_quantile_tiers()`) that
  takes the **existing, read-only** classifier risk scores and buckets them into 3 tiers by
  quantile thresholds (e.g. terciles, or proposal-specified cut points if any were given — confirm).
- Surfaced as an **alternate column/toggle**, clearly labeled "Quantile tiers (DA1-compliance
  demo)" next to the existing Critical/High/Medium/Low tiers — never overwriting or replacing them.
- Lowest-risk, no new dependencies, no pipeline coupling — pure post-hoc relabeling of existing scores.

### 7. Report-compliance checklist

- `docs/report_compliance_checklist.md`: a table — DA1 claim → proving file/output/demo screen.
  Written last, once all other outputs exist, so it can cite real paths (e.g. "RF/XGBoost count
  regression + MAE/RMSE/R² comparison → `results/regression/metrics.json`,
  `results/regression/comparison_table.md`").

## Execution order

**Phase 0 — setup (must happen first, blocking):**
- Add `xgboost`, `geopandas`, `shapely`, `folium` to dependencies (proposal: a new
  `requirements-report-compliance.txt` rather than editing root `requirements.txt`, so the frozen
  env's dependency set is visibly untouched — confirm this choice with you).
- Verify each installs cleanly in the Windows venv **before** writing code against it.

**Phase 1 — parallel, independent, low/self-contained risk:**
- **(A) Item 4** GeoPandas/Shapely grid rewrite + equivalence test — fully self-contained, only
  needs raw collision lat/lon.
- **(B) Item 6** Quantile tiers — trivial, only needs existing frozen scores (read-only).
- **(C) Item 2a** Collision-level temporal EDA — pure analysis, no pipeline dependency.

**Phase 2 — depends on Phase 1C, feeds Phase 3:**
- **(D) Item 2b** Grid-month temporal aggregate features (rush-hour share, weekend share, etc.)
  + their leakage tests.
- **(E) Item 5** Road-type/weather grid-month aggregate features, same feature table as (D).

**Phase 3 — the big one, depends on Phase 2 output:**
- **(F) Item 1** Regression track: new feature table (temporal + road/weather aggregates, plus
  spatial/lag-style features recomputed the same way the classifier does, since those can't be
  read from the frozen `ml_features.parquet` without coupling to it) → baseline/RF/XGBoost →
  new lock file → metrics/comparison table.

**Phase 4 — depends on Phase 3 (and 1E/1B for layers):**
- **(G) Item 3** Folium map: frozen classifier predictions (read-only) + regression predictions
  (Phase 3) + quantile tiers (1B) + road/weather filters (1E) as layers.

**Phase 5 — last:**
- **(H) Item 7** Compliance checklist, final cross-check against DA1 proposal text.

## Risks / what's highest-risk

1. **GeoPandas/Shapely/Fiona/GDAL install on Windows (Phase 0 + 1A)** — these have historically
   been painful to `pip install` on Windows without conda (native binary deps). If `pip install
   geopandas shapely` fails, may need conda/mamba or prebuilt wheels. This could block Phase 1A
   and, transitively, nothing else (1A is isolated), but should be resolved early since it's a
   pure infra risk, not a logic risk.
2. **Regression track's own lock/freeze semantics (Phase 3)** — must be a genuinely separate
   mechanism from `src/freeze_config.py`/`src/evaluate_test.py` so there is zero chance of an
   accidental double-run or overwrite touching the classifier's lock. Proposal: copy the *pattern*
   into a new `src/report_compliance/freeze_regression.py` / `evaluate_regression_test.py` rather
   than parameterizing the existing scripts.
3. **Historical-average baseline granularity (Phase 3)** — ambiguous in the proposal; needs your
   confirmation (grid-only average vs grid+month-of-year average) since it changes the baseline's
   difficulty and the interpretation of RF/XGBoost's improvement over it.
4. **What "identical grid-cell assignment" tolerance means (Phase 1A)** — floating point/CRS
   rounding could cause a handful of boundary-cell mismatches between the pyproj and GeoPandas
   implementations; plan is to report exact-match rate and treat >99.9% as "logically equivalent,"
   but flagging this as a judgment call.

## Decisions (confirmed)

1. Regression target transform: **log1p(count)**, fit/validated in log space, inverted with
   `expm1()` before computing/reporting MAE/RMSE/R² on the raw count scale.
2. Historical-average baseline: **per-grid, per-calendar-month mean**, computed from
   training-window history only (captures seasonality, e.g. this grid's typical January count).
3. Dependencies: new **`requirements-report-compliance.txt`** for `xgboost`, `geopandas`,
   `shapely`, `folium` — root `requirements.txt` stays untouched.
4. New-artifact locations as proposed: `data/processed/regression_features.parquet`,
   `models/regression/`, `results/regression/`, `outputs/report_compliance/`,
   `tests/report_compliance/`.
5. Quantile tiers: **terciles (33rd/66th percentile)** → LOW/MEDIUM/HIGH.

Implementation proceeds in the phase order above, starting with Phase 0 (dependency install/verify)
and Phase 1 (parallel: GeoPandas/Shapely grid rewrite, quantile tiers, temporal EDA).
