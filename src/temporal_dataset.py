import pandas as pd
from pathlib import Path


INPUT_PATH = Path(
    "data/interim/collisions_with_grid.csv"
)

OUTPUT_PATH = Path(
    "data/processed/grid_monthly_collisions.csv"
)


MIN_TOTAL_COLLISIONS = 3


def print_section(title):
    """Print a formatted section heading."""

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def load_grid_collision_data():
    """Load collision data with spatial grid assignments."""

    print_section(
        "STEP 1: LOADING GRID COLLISION DATA"
    )

    df = pd.read_csv(
        INPUT_PATH,
        low_memory=False
    )

    df["datetime"] = pd.to_datetime(
        df["datetime"]
    )

    print(
        f"Total collision records: {len(df):,}"
    )

    print(
        f"Unique spatial grids: "
        f"{df['grid_id'].nunique():,}"
    )

    return df


def create_temporal_features(df):
    """Create year and month features."""

    print_section(
        "STEP 2: CREATING TEMPORAL FEATURES"
    )

    df["year"] = df["datetime"].dt.year

    df["month"] = df["datetime"].dt.month

    df["year_month"] = (
        df["datetime"]
        .dt.to_period("M")
        .astype(str)
    )

    print(
        f"Date range: "
        f"{df['datetime'].min()} "
        f"to "
        f"{df['datetime'].max()}"
    )

    print(
        f"Total months: "
        f"{df['year_month'].nunique()}"
    )

    return df


def select_active_grids(df):
    """
    Select grids with sufficient historical
    collision activity for prediction.
    """

    print_section(
        "STEP 3: SELECTING ACTIVE SPATIAL GRIDS"
    )

    grid_counts = (
        df.groupby("grid_id")
        .size()
    )

    active_grids = (
        grid_counts[
            grid_counts >= MIN_TOTAL_COLLISIONS
        ]
        .index
    )

    filtered_df = df[
        df["grid_id"].isin(active_grids)
    ].copy()

    print(
        f"Minimum total collisions per grid: "
        f"{MIN_TOTAL_COLLISIONS}"
    )

    print(
        f"Active grids selected: "
        f"{len(active_grids):,}"
    )

    print(
        f"Original grids: "
        f"{df['grid_id'].nunique():,}"
    )

    print(
        f"Collisions retained: "
        f"{len(filtered_df):,}"
    )

    return filtered_df


def create_monthly_aggregation(df):
    """Aggregate collisions by grid and month."""

    print_section(
        "STEP 4: CREATING MONTHLY AGGREGATION"
    )

    monthly_df = (
        df.groupby(
            [
                "grid_id",
                "year_month"
            ]
        )
        .agg(
            collision_count=(
                "collision_index",
                "count"
            ),

            total_casualties=(
                "number_of_casualties",
                "sum"
            ),

            average_severity=(
                "collision_severity",
                "mean"
            ),

            average_speed_limit=(
                "speed_limit",
                "mean"
            ),

            grid_latitude=(
                "latitude",
                "mean"
            ),

            grid_longitude=(
                "longitude",
                "mean"
            )
        )
        .reset_index()
    )

    monthly_df["year_month"] = pd.to_datetime(
        monthly_df["year_month"]
    )

    print(
        f"Observed grid-month records: "
        f"{len(monthly_df):,}"
    )

    return monthly_df


def create_complete_grid_history(
    monthly_df
):
    """
    Create complete monthly histories
    including zero-collision months.
    """

    print_section(
        "STEP 5: CREATING COMPLETE TEMPORAL HISTORY"
    )

    all_months = pd.date_range(
        start=monthly_df["year_month"].min(),
        end=monthly_df["year_month"].max(),
        freq="MS"
    )

    grid_ids = monthly_df[
        "grid_id"
    ].unique()

    full_index = pd.MultiIndex.from_product(
        [
            grid_ids,
            all_months
        ],
        names=[
            "grid_id",
            "year_month"
        ]
    )

    complete_df = (
        monthly_df
        .set_index(
            [
                "grid_id",
                "year_month"
            ]
        )
        .reindex(full_index)
        .reset_index()
    )

    print(
        f"Complete grid-month records: "
        f"{len(complete_df):,}"
    )

    print(
        f"Expected months per grid: "
        f"{len(all_months)}"
    )

    return complete_df


def fill_missing_values(
    complete_df
):
    """Fill zero-collision periods."""

    print_section(
        "STEP 6: FILLING ZERO-COLLISION MONTHS"
    )

    before_missing = (
        complete_df[
            "collision_count"
        ]
        .isna()
        .sum()
    )

    print(
        f"Zero-collision months to fill: "
        f"{before_missing:,}"
    )

    complete_df["collision_count"] = (
        complete_df[
            "collision_count"
        ]
        .fillna(0)
    )

    complete_df["total_casualties"] = (
        complete_df[
            "total_casualties"
        ]
        .fillna(0)
    )

    grid_features = [
        "average_severity",
        "average_speed_limit",
        "grid_latitude",
        "grid_longitude"
    ]

    for column in grid_features:

        complete_df[column] = (
            complete_df
            .groupby("grid_id")[column]
            .transform(
                lambda x: x.ffill().bfill()
            )
        )

    complete_df["year"] = (
        complete_df["year_month"].dt.year
    )

    complete_df["month"] = (
        complete_df["year_month"].dt.month
    )

    complete_df["year_month"] = (
        complete_df["year_month"]
        .dt.strftime("%Y-%m")
    )

    print(
        "Missing values handled successfully."
    )

    return complete_df


def analyze_dataset(df):
    """Analyze final temporal dataset."""

    print_section(
        "STEP 7: ANALYZING FINAL DATASET"
    )

    print(
        f"Total records: {len(df):,}"
    )

    print(
        f"Unique grids: "
        f"{df['grid_id'].nunique():,}"
    )

    print(
        f"Unique months: "
        f"{df['year_month'].nunique()}"
    )

    print(
        "\nCollision count statistics:"
    )

    print(
        df["collision_count"]
        .describe()
    )

    zero_months = (
        df["collision_count"] == 0
    ).sum()

    print(
        f"\nZero-collision records: "
        f"{zero_months:,}"
    )

    print(
        f"Zero-collision percentage: "
        f"{zero_months / len(df) * 100:.2f}%"
    )


def save_dataset(df):
    """Save final spatio-temporal dataset."""

    print_section(
        "STEP 8: SAVING FINAL TEMPORAL DATASET"
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        "Dataset saved to:"
    )

    print(
        OUTPUT_PATH
    )


def main():

    print("\n")
    print("#" * 60)

    print(
        "TRAFFIC ACCIDENT HOTSPOT PREDICTION"
    )

    print(
        "SPATIO-TEMPORAL DATASET CREATION"
    )

    print("#" * 60)

    df = load_grid_collision_data()

    df = create_temporal_features(df)

    df = select_active_grids(df)

    monthly_df = (
        create_monthly_aggregation(df)
    )

    complete_df = (
        create_complete_grid_history(
            monthly_df
        )
    )

    complete_df = fill_missing_values(
        complete_df
    )

    analyze_dataset(
        complete_df
    )

    save_dataset(
        complete_df
    )

    print_section(
        "SPATIO-TEMPORAL DATASET COMPLETE"
    )

    print(
        "Complete grid-based temporal "
        "dataset created successfully."
    )


if __name__ == "__main__":
    main()