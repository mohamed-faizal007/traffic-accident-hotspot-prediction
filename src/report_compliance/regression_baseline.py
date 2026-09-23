"""Historical-average baseline for the regression track: per-grid,
per-calendar-month mean collision count, fit on TRAINING data only.

Predicts a target month's count as that grid's mean actual count in the same
calendar month across the training years (e.g. predicting a grid's January
2024 count from its January 2021/2022/2023 counts) -- captures seasonality
without looking at any val/test-period outcome.
"""

import numpy as np
import pandas as pd


def fit_historical_average_baseline(train_df, target_col="next_month_collisions"):
    """Returns a fitted lookup: dict with grid+calendar-month means, per-grid
    fallback means, and a global fallback mean, computed from `train_df` only."""
    t = train_df.copy()
    t["cal_month"] = pd.to_datetime(t["target_month"]).dt.month
    by_grid_month = t.groupby(["grid_id", "cal_month"])[target_col].mean()
    by_grid = t.groupby("grid_id")[target_col].mean()
    global_mean = float(t[target_col].mean())
    return {"by_grid_month": by_grid_month, "by_grid": by_grid, "global_mean": global_mean}


def predict_historical_average_baseline(fitted, df):
    """Apply a baseline fitted with fit_historical_average_baseline to any
    frame with grid_id and target_month columns. Falls back grid-month ->
    grid -> global mean when a (grid, calendar-month) combination was never
    observed in training."""
    cal_month = pd.to_datetime(df["target_month"]).dt.month
    key = pd.MultiIndex.from_arrays([df["grid_id"].to_numpy(), cal_month.to_numpy()])
    pred = fitted["by_grid_month"].reindex(key).to_numpy()

    missing = np.isnan(pred)
    if missing.any():
        grid_fallback = df["grid_id"].map(fitted["by_grid"]).to_numpy()
        pred = np.where(missing, grid_fallback, pred)

    still_missing = np.isnan(pred)
    if still_missing.any():
        pred = np.where(still_missing, fitted["global_mean"], pred)

    return pred
