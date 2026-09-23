"""Feature table for the new regression track: predicts next-month collision
COUNT per grid (not the binary hotspot label).

Reuses, read-only, the same generic building blocks the frozen classifier
uses -- src/features.py::build_feature_table (spatial + lag/rolling-count
features, and the ground-truth next_month_collisions/target_month columns)
and src/splits.py::assign_split (identical train/val/test discipline) --
plus the new grid-month temporal/road/weather aggregate shares from Phase 2
(report_compliance/grid_month_aggregates.py). None of this writes to, or
depends on the CONTENTS of, data/processed/ml_features.parquet; it is an
entirely separate table written to data/processed/regression_features.parquet.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DATA_DIR  # noqa: E402
from features import build_feature_table, FEATURE_COLUMNS  # noqa: E402
from splits import assign_split  # noqa: E402
from temporal_dataset import load_grid_collision_data, select_active_grids, build_monthly_table  # noqa: E402
from report_compliance.grid_month_aggregates import build_aggregate_features, SHARE_COLUMNS  # noqa: E402

OUTPUT_PATH = PROCESSED_DATA_DIR / "regression_features.parquet"

# Spatial/lag features (reused from the classifier's generic feature builder)
# plus the new grid-month temporal/road/weather aggregate shares.
REGRESSION_FEATURE_COLUMNS = FEATURE_COLUMNS + SHARE_COLUMNS

RAW_TARGET_COLUMN = "next_month_collisions"
LOG_TARGET_COLUMN = "log1p_next_month_collisions"


def build_regression_feature_table():
    """Assemble the full grid x month regression feature table with splits."""
    df = load_grid_collision_data()
    active = select_active_grids(df)
    monthly = build_monthly_table(df, active)

    features = build_feature_table(monthly)  # reused, read-only; classifier's generic builder

    collisions = df[df["grid_id"].isin(active)]
    aggregates = build_aggregate_features(collisions, monthly[["grid_id", "year_month"]])
    features = features.merge(aggregates, on=["grid_id", "year_month"], how="left", validate="one_to_one")

    features["split"] = assign_split(features)  # reused, read-only; identical train/val/test discipline
    features[LOG_TARGET_COLUMN] = np.log1p(features[RAW_TARGET_COLUMN])
    return features


def main():
    table = build_regression_feature_table()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUTPUT_PATH, index=False)
    print(f"Rows: {len(table):,}  columns: {len(table.columns)}")
    print(table["split"].value_counts())
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
