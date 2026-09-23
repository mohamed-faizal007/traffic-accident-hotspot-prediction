"""Grid-month temporal and road/weather AGGREGATE features for the new
regression track (DA1 proposal items: explicit temporal features "available
to the regression track", and road-type/weather signals).

Timing contract (same standard as src/features.py's historical_* columns):
  * Every feature for grid g at month t is a CUMULATIVE share over all of
    that grid's collisions in months <= t (inclusive of month t itself,
    exactly like collisions_lag_0 / historical_avg_collisions already do).
  * A grid-month with zero collisions ever recorded by month t gets the
    sentinel value NEVER_SEEN_SENTINEL (-1.0), which is outside the valid
    [0, 1] share range -- the same "distinct sentinel, not 0" pattern
    src/features.py uses for months_since_last_collision.

Column codings (source: DfT STATS19 road safety data guide, the documented
schema for this dataset -- see README.md "Data provenance"):
  * weather_conditions == 1  -> "Fine, no high winds" (the modal category,
    ~80% of rows here); adverse_weather is simply "not that".
  * road_surface_conditions == 1 -> "Dry"; wet_road is "not that".
  * road_type == 6 -> "Single carriageway" (the modal category, ~73% of
    rows here); single_carriageway_share is the share matching it.
  * speed_limit is a literal mph value already, no decoding needed;
    high_speed is speed_limit >= 40.
These are all coarse, defensible splits against the single dominant/most
legible category per column, chosen so correctness does not hinge on the
exact meaning of every less-common code.

This module is entirely new and read-only with respect to every frozen
classifier file. It does not import from, or get imported by, src/features.py,
src/temporal_dataset.py, src/train_validate.py, src/freeze_config.py, or
src/evaluate_test.py, and it writes only to new output paths under
data/processed/ and results/regression/.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from report_compliance.temporal_eda import TIME_OF_DAY_BINS  # noqa: E402

NEVER_SEEN_SENTINEL = -1.0

RUSH_HOUR_BUCKETS = {"morning_rush", "evening_rush"}
NIGHT_BUCKET = "night"

FLAG_COLUMNS = [
    "is_rush_hour",
    "is_weekend",
    "is_night",
    "is_adverse_weather",
    "is_wet_road",
    "is_single_carriageway",
    "is_high_speed",
]

SHARE_COLUMNS = [f"historical_{flag[3:]}_share" for flag in FLAG_COLUMNS]


def add_flags(df):
    """Add one 0/1 flag column per FLAG_COLUMNS entry to a per-collision frame.

    Expects columns: time (HH:MM string), day_of_week (STATS19 1=Sun..7=Sat),
    weather_conditions, road_surface_conditions, road_type, speed_limit.
    """
    out = df.copy()
    hour = pd.to_datetime(out["time"], format="%H:%M", errors="coerce").dt.hour
    time_of_day = pd.cut(hour, bins=TIME_OF_DAY_BINS,
                          labels=["night", "morning_rush", "midday", "evening_rush", "evening"], right=True)

    out["is_rush_hour"] = time_of_day.isin(RUSH_HOUR_BUCKETS).astype(int)
    out["is_weekend"] = out["day_of_week"].isin({1, 7}).astype(int)
    out["is_night"] = (time_of_day == NIGHT_BUCKET).astype(int)
    out["is_adverse_weather"] = (out["weather_conditions"] != 1).astype(int)
    out["is_wet_road"] = (out["road_surface_conditions"] != 1).astype(int)
    out["is_single_carriageway"] = (out["road_type"] == 6).astype(int)
    out["is_high_speed"] = (out["speed_limit"] >= 40).astype(int)
    return out


def _to_matrix(monthly, value):
    """Pivot a complete (grid_id, year_month) table to a (grids, months) matrix.
    Independent re-implementation of the identical helper in src/features.py,
    kept separate on purpose so this module never imports frozen pipeline code."""
    monthly = monthly.sort_values(["grid_id", "year_month"])
    n_months = monthly["year_month"].nunique()
    if len(monthly) != monthly["grid_id"].nunique() * n_months:
        raise ValueError("monthly table must contain every grid for every month")
    return monthly[value].to_numpy(dtype=np.float64).reshape(-1, n_months)


def build_aggregate_features(collisions, monthly_skeleton):
    """Build cumulative-historical share features per (grid_id, year_month).

    `collisions` : per-collision rows with grid_id, year_month, and the raw
      columns `add_flags` needs (time, day_of_week, weather_conditions,
      road_surface_conditions, road_type, speed_limit).
    `monthly_skeleton` : a complete (grid_id, year_month) frame covering
      every active grid for every month in range (fill_value=0 semantics),
      e.g. the grid_id/year_month columns of
      src.temporal_dataset.build_monthly_table's output.

    Returns monthly_skeleton with SHARE_COLUMNS appended.
    """
    flagged = add_flags(collisions)

    monthly_counts = (
        flagged.groupby(["grid_id", "year_month"])
        .agg(_collisions=("grid_id", "size"), **{f"_{c}": (c, "sum") for c in FLAG_COLUMNS})
    )

    skeleton = monthly_skeleton[["grid_id", "year_month"]].copy()
    full = skeleton.set_index(["grid_id", "year_month"])
    monthly_counts = monthly_counts.reindex(full.index, fill_value=0).reset_index()
    monthly_counts = monthly_counts.sort_values(["grid_id", "year_month"]).reset_index(drop=True)

    total_matrix = _to_matrix(monthly_counts, "_collisions")
    cum_total = np.cumsum(total_matrix, axis=1)

    out = monthly_counts[["grid_id", "year_month"]].copy()
    for flag, share_col in zip(FLAG_COLUMNS, SHARE_COLUMNS):
        flag_matrix = _to_matrix(monthly_counts, f"_{flag}")
        cum_flag = np.cumsum(flag_matrix, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            share = cum_flag / cum_total
        share = np.where(cum_total == 0, NEVER_SEEN_SENTINEL, share)
        out[share_col] = share.ravel()

    return out
