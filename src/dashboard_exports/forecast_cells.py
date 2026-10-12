"""Per-cell DESCRIPTIVE drivers for the Jan-2026 forecast cells.

Everything here is read straight from existing leakage-safe features (values at Dec-2025) and the
cells' own monthly counts; percentiles are computed from the data (share of the modelled cells with a
strictly lower value). These are descriptive facts, NOT a model attribution (no SHAP, no coefficients).
Also attaches the within-month LOW/MEDIUM/HIGH quantile tier of the frozen classifier's forecast
scores, using the same function as the 2025 table.

Reads the feature table's 'unlabeled' rows and the label-free collision_count column of 2025 months.
Writes only outputs/dashboard/forecast_cells_2026-01.csv (+ .meta.json).
"""

import datetime
import json

import numpy as np
import pandas as pd

try:
    from ._common import DASHBOARD_DIR, OUTPUTS_DIR, PROJECT_ROOT
except ImportError:  # run as a script
    from _common import DASHBOARD_DIR, OUTPUTS_DIR, PROJECT_ROOT

from report_compliance.quantile_tiers import assign_quantile_tiers

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "regression_features.parquet"
CLASSIFIER_FORECAST = OUTPUTS_DIR / "forecast_2026-01.csv"
OUT = DASHBOARD_DIR / "forecast_cells_2026-01.csv"
META = DASHBOARD_DIR / "forecast_cells_2026-01.meta.json"

DRIVER_COLUMNS = ["collisions_last_12_months", "months_since_last_collision", "historical_avg_collisions",
                  "historical_total_collisions", "historical_max_collisions", "collisions_lag_0"]
SHARE_COLUMNS = ["historical_rush_hour_share", "historical_weekend_share", "historical_night_share",
                 "historical_adverse_weather_share", "historical_wet_road_share",
                 "historical_single_carriageway_share", "historical_high_speed_share"]


def percentile_lower(values: pd.Series) -> pd.Series:
    """% of cells whose value is strictly lower (0 = lowest, <100)."""
    v = values.to_numpy()
    sorted_v = np.sort(v)
    return pd.Series(np.searchsorted(sorted_v, v, side="left") / len(v) * 100, index=values.index)


def main():
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    cur = pd.read_parquet(FEATURES_PATH, columns=["grid_id", "year_month", "split"] + DRIVER_COLUMNS + SHARE_COLUMNS,
                          filters=[("split", "==", "unlabeled")]).drop(columns="split")
    last_month = cur["year_month"].iloc[0]
    assert cur["year_month"].nunique() == 1
    cur = cur.drop(columns="year_month")

    hist_start = last_month - pd.DateOffset(months=11)
    hist = pd.read_parquet(FEATURES_PATH, columns=["grid_id", "year_month", "collision_count"],
                           filters=[("year_month", ">=", hist_start)])
    hist = hist[hist["year_month"] <= last_month].sort_values(["grid_id", "year_month"])
    assert hist.groupby("grid_id").size().eq(12).all(), "expected 12 monthly counts per cell"
    history = hist.groupby("grid_id")["collision_count"].apply(lambda s: ";".join(str(int(x)) for x in s))

    out = cur.merge(history.rename("history_12m"), on="grid_id", how="left", validate="one_to_one")
    check = out["history_12m"].map(lambda s: sum(int(x) for x in s.split(";")))
    mismatch = int((check != out["collisions_last_12_months"]).sum())

    out["pct_historical_avg"] = percentile_lower(out["historical_avg_collisions"])
    out["pct_last_12_months"] = percentile_lower(out["collisions_last_12_months"])

    fc = pd.read_csv(CLASSIFIER_FORECAST, usecols=["grid_id", "score", "target_month"])
    fc["quantile_tier"] = assign_quantile_tiers(fc["score"].to_numpy(), fc["target_month"].to_numpy())
    out = out.merge(fc[["grid_id", "quantile_tier"]], on="grid_id", how="left", validate="one_to_one")

    out = out.sort_values("grid_id")
    out.to_csv(OUT, index=False, float_format="%.5g")
    META.write_text(json.dumps({
        "kind": "descriptive, not a model attribution",
        "as_of_month": last_month.strftime("%Y-%m"),
        "n_cells": int(len(out)),
        "percentile_definition": "percent of the modelled cells whose value is strictly lower",
        "history_start": hist_start.strftime("%Y-%m"),
        "history_12m": f"monthly collision counts {hist_start:%Y-%m}..{last_month:%Y-%m}, ';'-separated",
        "history_vs_last12_feature_mismatches": mismatch,
        "quantile_tier": "within-month terciles of the frozen classifier's Jan-2026 scores",
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }, indent=2))
    print(f"{len(out):,} cells; history-vs-feature mismatches: {mismatch}")
    print(out.head(2).T.to_string())


if __name__ == "__main__":
    main()
