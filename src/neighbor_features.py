"""Neighbor-cell (Moore-8 adjacency) collision-rate features -- ablation only.

Additive and read-only with respect to every frozen file. `compute_neighbor_features`
is a pure function (no file I/O) so it can be unit-tested with synthetic data, the
same split as src/report_compliance/grid_month_aggregates.py::build_aggregate_features.
`load_all_cell_months` / `build_neighbor_feature_table` do the production I/O, reading
data/interim/collisions_with_grid.csv and data/interim/grid_cells.csv (both already
produced by src/spatial_grid.py) -- read-only, never written to.

Timing contract (same standard as src/features.py's historical_* columns): the
neighbor aggregate for cell g at month t uses only neighbor collision data from
months <= t (inclusive of month t itself -- the same cutoff every other feature
in this project already uses).

Signal propagated: each neighbor cell's own leakage-safe cumulative
historical_avg_collisions (historical_total_collisions / n_obs, months <= t),
computed over EVERY occupied grid cell in the dataset -- not just the "active"
subset (>= ACTIVE_MIN_COLLISIONS in the training period) used for modelling --
because a quiet neighbor is still real local signal even if it's too sparse to
model directly on its own.

Adjacency: Moore-8 neighborhood via the integer grid_ix/grid_iy cell indices
already assigned by src/spatial_grid.py -- the 8 cells at
(grid_ix +/- {-1,0,1}, grid_iy +/- {-1,0,1}), excluding the cell itself. A
neighbor position that is not an occupied cell (zero collisions ever recorded
anywhere in the dataset) does not exist and is excluded from both the
aggregate and neighbor_cell_count. A cell with zero existing neighbors gets
neighbor_avg_collisions_mean/max = 0.0 -- not a distinct sentinel, since "no
neighbor data available" and "neighbors with no history" are treated alike
for this feature's purpose.

Not imported by, and does not import, src/features.py, src/train_validate.py,
src/freeze_config.py, src/evaluate_test.py, src/temporal_dataset.py, or their
regression-track equivalents.
"""

import warnings

import numpy as np
import pandas as pd

from config import INTERIM_DATA_DIR, FIRST_MONTH, LAST_MONTH

NEIGHBOR_OFFSETS = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if not (dx == 0 and dy == 0)]
assert len(NEIGHBOR_OFFSETS) == 8

NEIGHBOR_FEATURES = ["neighbor_avg_collisions_mean", "neighbor_avg_collisions_max", "neighbor_cell_count"]


def load_all_cell_months():
    """Every occupied grid cell x every month in [FIRST_MONTH, LAST_MONTH], with grid_ix/grid_iy.

    Deliberately covers ALL occupied cells in collisions_with_grid.csv, not just the
    "active" subset temporal_dataset.py selects for modelling.
    """
    collisions = pd.read_csv(
        INTERIM_DATA_DIR / "collisions_with_grid.csv",
        usecols=["grid_id", "grid_ix", "grid_iy", "datetime"], low_memory=False,
    )
    collisions["datetime"] = pd.to_datetime(collisions["datetime"])
    collisions["year_month"] = collisions["datetime"].dt.to_period("M").dt.to_timestamp()

    cells = collisions.drop_duplicates("grid_id")[["grid_id", "grid_ix", "grid_iy"]]
    months = pd.date_range(FIRST_MONTH, LAST_MONTH, freq="MS")

    counts = collisions.groupby(["grid_id", "year_month"]).size().rename("collision_count")
    full = pd.MultiIndex.from_product([sorted(cells["grid_id"]), months], names=["grid_id", "year_month"])
    monthly = counts.reindex(full, fill_value=0).reset_index()
    return monthly.merge(cells, on="grid_id").sort_values(["grid_id", "year_month"]).reset_index(drop=True)


def compute_neighbor_features(monthly):
    """Pure computation: monthly needs grid_id, grid_ix, grid_iy, year_month, collision_count,
    complete for every (cell, month) pair -- same "complete lattice" contract as
    src/features.py::build_feature_table's `monthly` input.

    Returns one row per (grid_id, year_month) with NEIGHBOR_FEATURES.
    """
    monthly = monthly.sort_values(["grid_id", "year_month"]).reset_index(drop=True)
    n_months = monthly["year_month"].nunique()
    n_cells_check = monthly["grid_id"].nunique()
    if len(monthly) != n_cells_check * n_months:
        raise ValueError("monthly table must contain every (cell, month) pair")

    cell_meta = monthly.drop_duplicates("grid_id")[["grid_id", "grid_ix", "grid_iy"]].reset_index(drop=True)
    n_cells = len(cell_meta)
    months = monthly["year_month"].drop_duplicates().sort_values().to_numpy()

    counts = monthly["collision_count"].to_numpy(dtype=np.float64).reshape(n_cells, n_months)
    cum_total = np.cumsum(counts, axis=1)
    n_obs = np.arange(1, n_months + 1)
    avg = cum_total / n_obs  # leakage-safe cumulative historical_avg_collisions, per cell

    index_of = {(int(ix), int(iy)): i for i, (ix, iy) in enumerate(zip(cell_meta["grid_ix"], cell_meta["grid_iy"]))}
    ix_arr, iy_arr = cell_meta["grid_ix"].to_numpy(), cell_meta["grid_iy"].to_numpy()

    offsets_avg = np.full((len(NEIGHBOR_OFFSETS), n_cells, n_months), np.nan)
    for k, (dx, dy) in enumerate(NEIGHBOR_OFFSETS):
        neighbor_idx = np.array([index_of.get((ix + dx, iy + dy), -1) for ix, iy in zip(ix_arr, iy_arr)])
        exists = neighbor_idx >= 0
        offsets_avg[k, exists, :] = avg[neighbor_idx[exists], :]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN slice = isolated cell, handled below
        neighbor_count = np.sum(~np.isnan(offsets_avg), axis=0)
        mean = np.nan_to_num(np.nanmean(offsets_avg, axis=0), nan=0.0)
        max_ = np.nan_to_num(np.nanmax(offsets_avg, axis=0), nan=0.0)

    out = {
        "grid_id": np.repeat(cell_meta["grid_id"].to_numpy(), n_months),
        "year_month": np.tile(months, n_cells),
        "neighbor_avg_collisions_mean": mean.ravel(),
        "neighbor_avg_collisions_max": max_.ravel(),
        "neighbor_cell_count": neighbor_count.ravel().astype(int),
    }
    return pd.DataFrame(out)


def build_neighbor_feature_table():
    """Production entry point: read collisions_with_grid.csv, compute NEIGHBOR_FEATURES
    for every occupied cell x month in the project's full date range."""
    return compute_neighbor_features(load_all_cell_months())
