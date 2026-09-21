"""Build the complete grid x month collision table.

Leakage rules:
  * Active grids are chosen from the TRAINING period only (2021-2023).
    Grids that first become active later are not modelled (documented limitation).
  * Grid coordinates are the geometric CENTRE of each 500 m cell (grid_cells.csv),
    a fixed property of the cell that carries no collision information.
"""

import pandas as pd

from config import (
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    FIRST_MONTH,
    LAST_MONTH,
    TRAIN_END_MONTH,
    ACTIVE_MIN_COLLISIONS,
)

INPUT_PATH = INTERIM_DATA_DIR / "collisions_with_grid.csv"
OUTPUT_PATH = PROCESSED_DATA_DIR / "grid_monthly_collisions.csv"


def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def load_grid_collision_data():
    print_section("STEP 1: LOADING GRID COLLISION DATA")
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["year_month"] = df["datetime"].dt.to_period("M").dt.to_timestamp()
    print(f"Collisions: {len(df):,}   grids: {df['grid_id'].nunique():,}")
    return df


def training_period(df):
    """Collisions up to and including TRAIN_END_MONTH."""
    end = pd.Period(TRAIN_END_MONTH, "M").to_timestamp()
    return df[df["year_month"] <= end]


def select_active_grids(df):
    """Grids with >= ACTIVE_MIN_COLLISIONS collisions in the training period."""
    print_section("STEP 2: SELECTING ACTIVE GRIDS (TRAINING PERIOD ONLY)")
    counts = training_period(df).groupby("grid_id").size()
    active = counts[counts >= ACTIVE_MIN_COLLISIONS].index
    print(f"Grids with any collision (all years): {df['grid_id'].nunique():,}")
    print(f"Active grids (>= {ACTIVE_MIN_COLLISIONS} collisions in {FIRST_MONTH}..{TRAIN_END_MONTH}): {len(active):,}")
    return active


def grid_coordinates(active):
    """Cell-centre latitude/longitude for the active grids."""
    cells = pd.read_csv(INTERIM_DATA_DIR / "grid_cells.csv").set_index("grid_id")
    return cells.loc[active, ["grid_latitude", "grid_longitude"]]


def build_monthly_table(df, active):
    """Complete grid x month table with explicit zero months."""
    print_section("STEP 3: MONTHLY AGGREGATION WITH ZERO MONTHS")
    sub = df[df["grid_id"].isin(active)]
    monthly = (
        sub.groupby(["grid_id", "year_month"])
        .agg(
            collision_count=("collision_index", "count"),
            total_casualties=("number_of_casualties", "sum"),
        )
    )
    months = pd.date_range(FIRST_MONTH, LAST_MONTH, freq="MS")
    full = pd.MultiIndex.from_product([sorted(active), months], names=["grid_id", "year_month"])
    monthly = monthly.reindex(full, fill_value=0).reset_index()
    monthly = monthly.merge(grid_coordinates(list(active)), left_on="grid_id", right_index=True)
    monthly["year"] = monthly["year_month"].dt.year
    monthly["month"] = monthly["year_month"].dt.month
    print(f"Rows: {len(monthly):,}  grids: {len(active):,}  months: {len(months)}")
    print(f"Zero-collision share: {(monthly['collision_count'] == 0).mean():.2%}")
    return monthly


def main():
    df = load_grid_collision_data()
    active = select_active_grids(df)
    monthly = build_monthly_table(df, active)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
