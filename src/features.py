"""Leakage-safe spatio-temporal features for grid-month hotspot prediction.

Timing contract
---------------
A row describes grid g at "prediction time" = end of month t.
  * every feature uses collision data from months <= t only
  * the label is whether month t+1 has >= HOTSPOT_MIN_COLLISIONS collisions

`delay` is only used for the alignment ablation: delay=1 reproduces the ORIGINAL
project alignment, where month t's own count was invisible to the features
(features saw months <= t-1 but the label was still month t+1).
"""

import numpy as np
import pandas as pd

from config import HOTSPOT_MIN_COLLISIONS, WARMUP_MONTHS

# `year` and the raw `month` are deliberately NOT features: year is out of range at test
# time, and month is fully described by month_sin / month_cos. The rolling means and
# last-3-month sum were exact rescalings/duplicates of kept window sums and were removed.
FEATURE_COLUMNS = [
    "grid_latitude",
    "grid_longitude",
    "month_sin",
    "month_cos",
    "collisions_lag_0",
    "collisions_lag_1",
    "collisions_lag_2",
    "casualties_lag_0",
    "collisions_rolling_sum_3",
    "historical_total_collisions",
    "historical_avg_collisions",
    "historical_max_collisions",
    "historical_collision_frequency",
    "historical_hotspot_frequency",
    "collisions_last_6_months",
    "collisions_last_12_months",
    "months_since_last_collision",
]

# Static per-grid features from DBSCAN clusters fitted on training-period
# collisions only (see hotspot_detection.py).
CLUSTER_FEATURES = ["grid_in_cluster", "cluster_collision_density"]
FEATURE_COLUMNS_WITH_CLUSTER = FEATURE_COLUMNS + CLUSTER_FEATURES

TARGET_COLUMN = "hotspot"


def _to_matrix(monthly, value):
    """Pivot a complete (grid_id, year_month) table to a (grids, months) matrix."""
    monthly = monthly.sort_values(["grid_id", "year_month"])
    n_months = monthly["year_month"].nunique()
    if len(monthly) != monthly["grid_id"].nunique() * n_months:
        raise ValueError("monthly table must contain every grid for every month")
    return monthly[value].to_numpy(dtype=np.float64).reshape(-1, n_months)


def _shift_right(matrix, delay):
    """Make month m see the value of month m-delay; earlier months are unknown (NaN)."""
    if delay == 0:
        return matrix
    out = np.full_like(matrix, np.nan)
    out[:, delay:] = matrix[:, :-delay]
    return out


def _window_sum(values, window):
    """Sum of the last `window` visible months, inclusive of the current one."""
    cs = np.concatenate([np.zeros((values.shape[0], 1)), np.cumsum(values, axis=1)], axis=1)
    idx = np.arange(1, values.shape[1] + 1)
    return cs[:, idx] - cs[:, np.maximum(idx - window, 0)]


def _lag(values, k):
    out = np.zeros_like(values)
    if k == 0:
        return values.copy()
    out[:, k:] = values[:, :-k]
    return out


def build_feature_table(monthly, delay=0, hotspot_min=HOTSPOT_MIN_COLLISIONS, warmup=WARMUP_MONTHS,
                        cluster_features=None):
    """Return one row per (grid, month t) with features, target and baseline columns.

    `monthly` needs: grid_id, year_month, collision_count, total_casualties,
    grid_latitude, grid_longitude. The last month has no label (target NaN); it is
    kept so it can be used for genuine forecasts. `cluster_features` (optional) is a
    per-grid table [grid_id, *CLUSTER_FEATURES] built from training-period data only;
    grids missing from it get 0 (not in any cluster).
    """
    monthly = monthly.sort_values(["grid_id", "year_month"]).reset_index(drop=True)
    grids = monthly["grid_id"].unique()
    months = np.sort(monthly["year_month"].unique())
    n_grids, n_months = len(grids), len(months)

    counts = _to_matrix(monthly, "collision_count")
    casualties = _to_matrix(monthly, "total_casualties")

    visible = _shift_right(counts, delay)          # collision history visible at month t
    visible_cas = _shift_right(casualties, delay)
    known = ~np.isnan(visible)
    v = np.nan_to_num(visible)
    v_cas = np.nan_to_num(visible_cas)

    n_obs = np.maximum(np.cumsum(known, axis=1), 1)
    idx = np.arange(n_months)[None, :].repeat(n_grids, axis=0)

    total = np.cumsum(v, axis=1)
    last_seen = np.maximum.accumulate(np.where(v > 0, idx, -1), axis=1)
    months_since = np.where(last_seen >= 0, idx - last_seen, n_obs)  # never seen -> n_obs

    sum3, sum6, sum12 = (_window_sum(v, w) for w in (3, 6, 12))

    features = {
        "collisions_lag_0": v,
        "collisions_lag_1": _lag(v, 1),
        "collisions_lag_2": _lag(v, 2),
        "casualties_lag_0": v_cas,
        "collisions_rolling_sum_3": sum3,
        "historical_total_collisions": total,
        "historical_avg_collisions": total / n_obs,
        "historical_max_collisions": np.maximum.accumulate(v, axis=1),
        "historical_collision_frequency": np.cumsum(v > 0, axis=1) / n_obs,
        "historical_hotspot_frequency": np.cumsum(v >= hotspot_min, axis=1) / n_obs,
        "collisions_last_6_months": sum6,
        "collisions_last_12_months": sum12,
        "months_since_last_collision": months_since,
    }

    # Target and baselines always use the TRUE counts, independent of `delay`.
    next_count = np.full_like(counts, np.nan)
    next_count[:, :-1] = counts[:, 1:]

    ts = pd.DatetimeIndex(months)
    table = {
        "grid_id": np.repeat(grids, n_months),
        "year_month": np.tile(ts, n_grids),
        "month_index": np.tile(np.arange(n_months), n_grids),
        "collision_count": counts.ravel(),
        "next_month_collisions": next_count.ravel(),
        "base_last_month_count": counts.ravel(),
        "base_trailing_12": _window_sum(counts, 12).ravel(),
    }
    for name, matrix in features.items():
        table[name] = matrix.ravel()
    df = pd.DataFrame(table)

    coords = monthly.drop_duplicates("grid_id").set_index("grid_id")[["grid_latitude", "grid_longitude"]]
    df["grid_latitude"] = df["grid_id"].map(coords["grid_latitude"])
    df["grid_longitude"] = df["grid_id"].map(coords["grid_longitude"])
    df["year"] = df["year_month"].dt.year
    df["month"] = df["year_month"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["target_month"] = df["year_month"] + pd.offsets.MonthBegin(1)
    df[TARGET_COLUMN] = np.where(
        df["next_month_collisions"].isna(), np.nan, (df["next_month_collisions"] >= hotspot_min).astype(float)
    )

    if cluster_features is not None:
        df = df.merge(cluster_features[["grid_id", *CLUSTER_FEATURES]], on="grid_id", how="left")
        df[CLUSTER_FEATURES] = df[CLUSTER_FEATURES].fillna(0)

    df = df[df["month_index"] >= warmup].reset_index(drop=True)
    return df
