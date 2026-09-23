# Neighbor-Cell Features: Ablation Plan

Branch `neighbor-features`, off `main` (the report-compliance track was already
merged into `main` at `2aa5b5a`, so this branches directly off `main`, not a
separate `report-compliance` branch).

**Goal.** Test whether adding spatially-adjacent-cell information improves
either the frozen classifier or the regression track. This is a named,
honestly-documented limitation of both systems: every existing feature in
`src/features.py` and `src/report_compliance/grid_month_aggregates.py` is
per-grid-cell only — no cell ever sees what is happening one door down. This
plan proposes an additive ablation, mirroring the rigor of the existing
DBSCAN cluster-feature ablation (`tests/test_cluster_features.py`,
`src/validation_comparison.py::cluster_ablation`) and the regression track's
own bootstrap-CI selection machinery (`regression_evaluation.py`).

**Nothing frozen is touched.** No code in this plan reads or writes any of:
`results/frozen_config.json`, `results/metrics.json`,
`results/test_evaluated.lock`, `models/final_model.joblib`,
`models/calibrator.joblib`, `data/processed/ml_features.parquet`,
`results/regression/regression_frozen_config.json`,
`results/regression/regression_test_evaluated.lock`,
`models/regression/final_model.joblib`. See "File I/O boundaries" below for
the exact new paths this work would write to instead — including two files
(`results/metrics.json`'s regression-track sibling `regression_metrics.json`,
and `data/processed/regression_features.parquet`) that are not in the
protected list above but that I'm proposing to also leave untouched, for the
same reason.

---

## 1. Feature design

### Adjacency definition

`src/spatial_grid.py::assign_grid` already assigns every collision an
integer `grid_ix`, `grid_iy` (BNG-metre cell index) and
`grid_id = f"{grid_ix}_{grid_iy}"`; `data/interim/grid_cells.csv` has one row
per occupied cell with those same integers. This makes adjacency exact and
free of any distance math: **Moore neighborhood, the 8 cells at
`(grid_ix ± {-1,0,1}, grid_iy ± {-1,0,1})` excluding the cell itself.**
I'm proposing this over a radius-based join (e.g. everything within 500 m of
the cell centre via `geopandas.sjoin`, echoing
`src/report_compliance/spatial_grid_geopandas.py`) because it's simpler,
deterministic, and for a regular 500 m square grid a fixed-radius circle and
the Moore-8 set are nearly the same cells anyway — the extra machinery
wouldn't buy much. Flagging this as a choice, not a certainty.

### Which existing signal to propagate from neighbors

Propose reusing **`historical_avg_collisions`** — already a leakage-safe,
cumulative, per-cell rate (`historical_total_collisions / n_obs`, defined in
`src/features.py::build_feature_table`) — computed independently **per
neighbor cell**, then aggregated across whichever of the up to 8 neighbors
exist. This is the most information-dense single existing column (it's
already a smoothed rate, not a noisy single-month count), and reusing it
avoids inventing a second definition of "collision rate" to keep in sync.

### Which cells count as "neighbors" — active grids only, or all grids?

The classifier and regression track both model only **active** grids (≥3
collisions in the training period, `select_active_grids` in
`src/temporal_dataset.py`). A geographically adjacent cell can easily fail
that bar — e.g. a quiet residential street next to a busy junction — but its
low-but-nonzero history is still real local signal the center cell's own
features can't see. **Proposing: compute neighbor collision history from
ALL occupied grid cells** (every `grid_id` present in
`data/interim/collisions_with_grid.csv`, not just the active subset used for
modelling), read directly from that file plus `grid_cells.csv` for the
`grid_ix`/`grid_iy` lookup — independent of, and not modifying,
`temporal_dataset.py`'s active-grid selection. A neighbor cell with zero
recorded collisions ever (present in the study's spatial extent but never
occupied) is simply absent from the collision-history matrix and is treated
as `historical_avg_collisions = 0` for that neighbor — a real, legitimate
zero (not a "never seen" sentinel case, unlike `months_since_last_collision`
in `features.py`, since 0 collisions/month is a valid average, not a missing
value).

### Aggregation and the resulting feature columns

Proposing three new columns, computed for each `(grid_id, year_month)`:

| column | definition |
|---|---|
| `neighbor_avg_collisions_mean` | mean of `historical_avg_collisions` over the existing neighbors (0 if a neighbor cell has no recorded history; the *cell itself* still counts as "existing" if it's ever occupied — see edge case below) |
| `neighbor_avg_collisions_max` | max of the same — "is there one clearly bad neighbor," a signal `mean` alone would blur |
| `neighbor_cell_count` | how many of the 8 possible neighbor positions are occupied cells at all (1–8) — mean-of-2 and mean-of-8 have very different variance, and this lets the model see that rather than guess it |

I'm proposing to test this as **one bundled block**, not feature-by-feature,
the same way `CLUSTER_FEATURES = ["grid_in_cluster", "cluster_collision_density"]`
is ablated as a pair rather than separately in
`validation_comparison.py::cluster_ablation` — one pre-registered
"neighbor features help y/n" decision, not three separate hypothesis tests
inflating the chance of a spurious positive. Open question for you: would
you rather I test `mean` alone first (leaner, matches the project's stated
preference for dropping redundant/low-value columns — see the comment at
the top of `src/features.py` about removing rescaled duplicates), and only
add `max`/`neighbor_cell_count` if that alone doesn't move the needle?

`sum` was considered and rejected as a fourth candidate: it conflates "one
busy neighbor" with "many mildly busy neighbors" and is highly collinear
with `neighbor_cell_count × mean`, so it adds little beyond what `mean` and
`max` already separately capture.

---

## 2. Leakage safety

The neighbor aggregate at grid `g`, month `t` is a function of
`historical_avg_collisions` for `g`'s neighbors, each of which — like every
other `historical_*` column in `features.py` — is a **cumulative** figure
over neighbor months `≤ t` (inclusive of month `t` itself, the same cutoff
the rest of the project already uses: `historical_avg_collisions` for the
cell *itself* also includes month `t`). So neighbor data as of month `t` is
exactly as "current" as every other feature at month `t` — this is not a
new leakage surface, just the same cutoff applied to eight more cells per
row.

Proposed new test file: `tests/test_neighbor_features_leakage.py`, same
synthetic-grid method as `tests/test_features_leakage.py` and
`tests/report_compliance/test_grid_month_aggregates_leakage.py`:

1. **`test_neighbor_features_ignore_future_months`** — build a small
   synthetic grid (e.g. a 3×3 block of `grid_ix,grid_iy`, collisions only at
   the center cell's neighbors), compute neighbor features, then inject a
   large collision spike into a neighbor cell strictly *after* month `t` and
   confirm month `t`'s neighbor features are unchanged (mirrors
   `test_features_ignore_future_months`).
2. **`test_neighbor_features_do_depend_on_neighbor_current_month`** — sanity
   check in the other direction: a collision added to a neighbor cell *in*
   month `t` DOES change the center cell's month-`t` neighbor features
   (mirrors `test_features_do_depend_on_current_month`). This is the
   assertion that actually proves the cutoff is `t`, not `t-1`.
3. **`test_hand_computed_neighbor_aggregate`** — a fully hand-worked small
   example (2–3 neighbors with known `historical_avg_collisions` at a fixed
   month) checking `neighbor_avg_collisions_mean`, `_max`, and
   `neighbor_cell_count` against arithmetic done by hand, the same pattern
   as `test_hand_computed_values` / `test_hand_computed_neighbor_aggregate`
   in the two existing leakage test files.
4. **`test_edge_of_grid_has_fewer_neighbors`** — a cell at the edge of the
   occupied extent (fewer than 8 occupied neighbor positions) gets a
   `neighbor_cell_count < 8` and its `mean`/`max` are computed only over the
   neighbors that exist, not padded with phantom zeros that would silently
   bias `mean` toward 0 for every edge cell (edge cells are not rare in this
   dataset's irregular occupied footprint, so getting this wrong would be a
   real, not theoretical, bug).
5. **`test_isolated_cell_has_zero_neighbor_count`** — a synthetic cell with
   no occupied neighbors at all gets `neighbor_cell_count == 0` and
   `neighbor_avg_collisions_mean/max` fall back to a defined value (proposing
   `0.0`, since "no neighbors" and "neighbors with zero history" are
   observationally similar for this feature's purpose — flag if you'd rather
   this be a distinct sentinel like `NEVER_SEEN_SENTINEL` in
   `grid_month_aggregates.py`).

---

## 3. Evaluation protocol

Fair, apples-to-apples, and reusing existing frozen validation numbers
rather than re-running everything:

### Classifier ablation

- Model family: **`HistGradientBoostingClassifier`**, the one actually
  frozen (`results/metrics.json["selection"]["model_family"] ==
  "hist_gradient_boosting"`, params `learning_rate=0.03, max_leaf_nodes=8,
  min_samples_leaf=50` — read from `results/metrics.json`, **read-only**,
  never rewritten).
- Split: identical `train` (target months ≤ 2023-12) / `val` (2024) frames
  from `data/processed/ml_features.parquet`, loaded read-only via
  `train_validate.py::load_split_frames`, with the new neighbor columns
  **joined in memory** for this script only — never written back into
  `ml_features.parquet`.
- Feature sets compared: `FEATURE_COLUMNS` (existing, 17 columns) vs
  `FEATURE_COLUMNS + NEIGHBOR_FEATURES` (+3 columns).
- Same 5-seed protocol as `cluster_ablation` (`src/validation_comparison.py`
  lines 31–59): reuse `sample_training`/`fit_hgb`/`score` from
  `train_validate.py` unmodified, varying only the negative-sampling seed,
  fitting both feature sets per seed, recording validation AP (+ top-1%/5%
  precision) for each.
- Statistical test: extend `validation_comparison.py::bootstrap`'s
  `comparisons` list with
  `("hist_gradient_boosting_with_neighbor", "hist_gradient_boosting")`,
  same grid-resampled 300-iteration bootstrap, 95% CI on the AP difference.

### Regression ablation

- Model family: **XGBoost**, the frozen regression choice
  (`n_estimators=300, max_depth=4, learning_rate=0.05`, read from
  `results/regression/regression_frozen_config.json`, **read-only**).
- Split: `train`/`val` frames from
  `report_compliance/regression_features.py::build_regression_feature_table()`
  (target ≤2023 / 2024), with neighbor columns joined in memory, not
  persisted to `data/processed/regression_features.parquet`.
- Feature sets: `REGRESSION_FEATURE_COLUMNS` (24 columns) vs `+
  NEIGHBOR_FEATURES`.
- A handful of `random_state` seeds (proposing 3, since XGBoost's own fit is
  otherwise deterministic given fixed data — there's no negative-sampling
  step here unlike the classifier, so seed variance is smaller and doesn't
  need 5).
- Statistical test: reuse `regression_evaluation.py::paired_bootstrap_mae_diff`
  directly (already generic over any two prediction arrays + grid ids) for
  `with_neighbor` vs `without`. Also report RMSE/R² deltas for context, but
  MAE stays the decision metric (matching the pre-registered regression
  selection rule's own metric choice).

### File I/O boundaries for this step

New, additive outputs only:
- `src/neighbor_features.py` — the feature builder (shared by both ablations)
- `results/neighbor_features_ablation.json` — classifier ablation results
  (NOT written into `results/metrics.json`, which is on the protected list)
- `results/regression/neighbor_features_ablation.json` — regression ablation
  results (proposing to also leave `results/regression/regression_metrics.json`
  untouched even though it isn't on your explicit protected list, for
  consistency with "additive, not destructive" and because that file backs
  claims in `docs/report_compliance_checklist.md`)
- `tests/test_neighbor_features_leakage.py`

---

## 4. Decision rule — my recommendation

**Pre-registered rule (to write down before running anything, same
discipline as `register_regression_selection_rule.py`):** neighbor features
are a genuine finding for a track if the 95% grid-bootstrap CI of the
validation metric difference (`with_neighbor − without`, AP for the
classifier / MAE for regression, sign flipped for MAE since lower is better)
excludes 0 in the improving direction.

**On whether to then also burn a one-shot test-set evaluation: I recommend
NOT doing it in this first pass, even if the validation CI excludes 0.**
Reasons:

1. The regression track's own item-3 result (XGBoost essentially tying the
   baseline on test MAE despite a clear validation-time edge) is a direct,
   recent demonstration that a validation win on this data doesn't reliably
   survive to test — exactly the generalization gap the "evaluate once"
   discipline exists to surface honestly, not to paper over with a second
   spend of that same one-shot budget on a feature-ablation whim.
2. A one-shot test evaluation for either track means building an entire new
   parallel freeze apparatus (its own `frozen_config.json`-equivalent
   pre-registered *before* looking at the ablation's validation numbers, its
   own `.lock` file, its own frozen model artifact) — real infrastructure
   work, not a quick follow-up, and premature if the validation gain turns
   out to be small.
3. Test-set evaluations are precious specifically because they can only
   happen once per track, ever. Spending that on a "does this incremental
   feature help" question — rather than reserving it for a change the
   validation evidence says is clearly, substantially better — is the kind
   of decision that's easy to regret and impossible to undo.

**Concrete bar I'd apply before even considering a follow-on test
evaluation:** the CI must exclude 0 *with margin* (not just barely — e.g. CI
lower bound at least ~0.01 AP away from 0, given the classifier's own
test-set AP is 0.2549 against a 2.34% base rate, so a marginal 0.002 AP bump
is noise-adjacent even if technically CI-positive) AND the point estimate
should be large enough to plausibly change how the system is described in
the viva, not just nudge a footnote. If validation clearly clears that bar,
I'd come back to you with the specific numbers before writing any freeze
infrastructure — that's a separate decision point, not something to fold
into this branch's first pass.

**If you'd rather set a different bar, or want the one-shot test queued up
regardless of how strong the validation signal is, tell me and I'll adjust
before writing any code.**

---

## 5. Computational cost and proposed order

**Cost.** Building the neighbor-history matrix reads
`collisions_with_grid.csv` (513,748 rows) and `grid_cells.csv` once, does
vectorized cumulative-sum arithmetic per cell (same style as
`features.py`/`grid_month_aggregates.py`, both already fast on this data),
then a `grid_ix,grid_iy` self-join for the 8 neighbor offsets — seconds, not
minutes, even over all occupied cells (not just the active subset).
Model refits: HistGradientBoosting × 2 feature sets × 5 seeds = 10 fits,
each using early stopping (`max_iter=400`, `n_iter_no_change=20`) — the
existing `train_validate.py` full tuning sweep (dozens of HGB configs) runs
in a couple of minutes, so 10 fits of one fixed config is well under a
minute. XGBoost × 2 feature sets × 3 seeds = 6 fits, similarly fast (the
existing `XGB_GRID` sweep in `train_validate_regression.py` has 4 configs
and isn't the bottleneck in that script). The 300-iteration bootstrap is
pure resampling of already-computed scores, no refitting — cheap regardless
of track. **Total: low single-digit minutes**, dominated by script startup
and I/O, not model fitting.

**Proposed order: classifier first, then regression, sequentially (not
parallel).**
- The classifier is the primary, production system — the named limitation
  matters more there, and it's the one referenced in the DA1 checklist's
  "Honest scope note" as the system this project actually stands behind.
- The neighbor-feature builder (`src/neighbor_features.py`) is shared code
  for both ablations; building and leakage-testing it once against the
  classifier's simpler feature set first, then reusing it as-is for the
  regression track, is lower-risk than debugging both integrations at once.
- Doing the classifier pass first also produces a concrete validation
  AP-diff number that tells us something about whether neighbor
  autocorrelation matters *at all* for this data before spending a second
  round of compute finding out the same thing on the regression track's
  much sparser target (83.8% zero grid-months, per the checklist) — if
  neighbor features show no signal for the (denser, better-behaved)
  classification target, that's useful context before also testing them on
  the harder count-regression target.
- If you'd rather see both results at once (e.g. to write up both findings
  together for the viva), I can run them in parallel instead — flag if so.

---

## Open questions for you before I write any code

1. **Adjacency:** Moore-8 via `grid_ix`/`grid_iy` (proposed) vs a
   radius-based `geopandas.sjoin`?
2. **Neighbor pool:** all occupied cells including inactive/low-volume ones
   (proposed) vs active grids only?
3. **Feature set:** `mean` + `max` + `neighbor_cell_count` bundle (proposed)
   vs `mean` alone first, adding the others only if that doesn't move the
   needle?
4. **Isolated-cell fallback:** plain `0.0` (proposed) vs a distinct
   sentinel value like `grid_month_aggregates.py`'s `NEVER_SEEN_SENTINEL`?
5. **One-shot test-set evaluation:** agree to stop at the validation-level
   ablation unless the gain clearly clears the margin described in section 4
   (my recommendation), or set a different bar / decide now regardless of
   outcome?
6. **Order:** classifier ablation first then regression (proposed,
   sequential) vs both in parallel?

Waiting for your go-ahead (and answers to whichever of the above you want to
weigh in on — I'll use the proposed defaults for anything you don't) before
writing `src/neighbor_features.py` or any tests.
