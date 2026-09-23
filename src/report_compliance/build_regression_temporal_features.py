"""Build the grid-month temporal/road/weather aggregate feature table for the
new regression track, over the real (active-grid) dataset.

Reuses src/temporal_dataset.py's load_grid_collision_data() and
select_active_grids() / build_monthly_table() as read-only function calls --
this only reads data/interim/collisions_with_grid.csv and never writes to
data/processed/grid_monthly_collisions.csv (that file, and the frozen
data/processed/ml_features.parquet, are untouched). Output goes to a new
file: data/processed/regression_temporal_features.parquet.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DATA_DIR  # noqa: E402
from temporal_dataset import load_grid_collision_data, select_active_grids, build_monthly_table  # noqa: E402
from report_compliance.grid_month_aggregates import build_aggregate_features, SHARE_COLUMNS  # noqa: E402

OUTPUT_PATH = PROCESSED_DATA_DIR / "regression_temporal_features.parquet"


def main():
    df = load_grid_collision_data()
    active = select_active_grids(df)
    monthly = build_monthly_table(df, active)  # read-only call; writes nothing itself

    collisions = df[df["grid_id"].isin(active)]
    aggregates = build_aggregate_features(collisions, monthly[["grid_id", "year_month"]])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    aggregates.to_parquet(OUTPUT_PATH, index=False)
    print(f"Rows: {len(aggregates):,}  grids: {aggregates['grid_id'].nunique():,}")
    print(aggregates[SHARE_COLUMNS].describe())
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
