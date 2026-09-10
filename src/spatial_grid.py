import pandas as pd
import numpy as np

from config import CLEAN_COLLISIONS_PATH


GRID_SIZE_KM = 0.5


def print_section(title):
    """Print a formatted section heading."""

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def load_collision_data():
    """Load cleaned collision data."""

    print_section(
        "STEP 1: LOADING CLEAN COLLISION DATA"
    )

    df = pd.read_csv(
        CLEAN_COLLISIONS_PATH,
        low_memory=False
    )

    df["datetime"] = pd.to_datetime(
        df["datetime"]
    )

    print(
        f"Total collisions loaded: {len(df):,}"
    )

    print(
        f"Total columns loaded: "
        f"{len(df.columns)}"
    )

    return df


def create_spatial_grid(df):
    """
    Assign collisions to approximately
    500 metre geographic grid cells.
    """

    print_section(
        "STEP 2: CREATING SPATIAL GRID"
    )

    # Approximate conversion
    # 1 degree latitude ≈ 111 km

    grid_size_degrees = (
        GRID_SIZE_KM / 111
    )

    print(
        f"Grid size: {GRID_SIZE_KM} km"
    )

    print(
        f"Grid size in degrees: "
        f"{grid_size_degrees:.6f}"
    )

    # Create grid coordinates

    df["grid_lat"] = (
        np.floor(
            df["latitude"]
            / grid_size_degrees
        )
        * grid_size_degrees
    )

    df["grid_lon"] = (
        np.floor(
            df["longitude"]
            / grid_size_degrees
        )
        * grid_size_degrees
    )

    # Create unique grid ID

    df["grid_id"] = (
        df["grid_lat"]
        .round(6)
        .astype(str)
        + "_"
        + df["grid_lon"]
        .round(6)
        .astype(str)
    )

    total_grids = (
        df["grid_id"]
        .nunique()
    )

    print(
        f"Total spatial grids created: "
        f"{total_grids:,}"
    )

    return df


def analyze_grid_distribution(df):
    """Analyze collisions across grids."""

    print_section(
        "STEP 3: ANALYZING GRID DISTRIBUTION"
    )

    grid_counts = (
        df["grid_id"]
        .value_counts()
    )

    print(
        "Grid collision statistics:"
    )

    print(
        f"Minimum collisions: "
        f"{grid_counts.min()}"
    )

    print(
        f"Maximum collisions: "
        f"{grid_counts.max()}"
    )

    print(
        f"Average collisions: "
        f"{grid_counts.mean():.2f}"
    )

    print(
        f"Median collisions: "
        f"{grid_counts.median():.2f}"
    )

    print(
        "\nTop 10 accident grids:"
    )

    print(
        grid_counts
        .head(10)
    )

    return grid_counts


def create_grid_summary(df):
    """Create accident statistics for each grid."""

    print_section(
        "STEP 4: CREATING GRID SUMMARY"
    )

    grid_summary = (
        df
        .groupby("grid_id")
        .agg(
            collision_count=(
                "collision_index",
                "count"
            ),
            average_latitude=(
                "latitude",
                "mean"
            ),
            average_longitude=(
                "longitude",
                "mean"
            ),
            total_casualties=(
                "number_of_casualties",
                "sum"
            ),
            average_severity=(
                "collision_severity",
                "mean"
            ),
            first_collision=(
                "datetime",
                "min"
            ),
            last_collision=(
                "datetime",
                "max"
            )
        )
        .reset_index()
    )

    grid_summary = (
        grid_summary
        .sort_values(
            "collision_count",
            ascending=False
        )
    )

    print(
        f"Total grid records: "
        f"{len(grid_summary):,}"
    )

    print(
        "\nTop 10 grids:"
    )

    print(
        grid_summary
        .head(10)
        .to_string(index=False)
    )

    return grid_summary


def save_grid_data(
    df,
    grid_summary
):
    """Save spatial grid datasets."""

    print_section(
        "STEP 5: SAVING SPATIAL GRID DATA"
    )

    collision_output = (
        "data/interim/collisions_with_grid.csv"
    )

    summary_output = (
        "data/interim/grid_summary.csv"
    )

    df.to_csv(
        collision_output,
        index=False
    )

    grid_summary.to_csv(
        summary_output,
        index=False
    )

    print(
        "Collision grid dataset saved:"
    )

    print(
        collision_output
    )

    print(
        "\nGrid summary saved:"
    )

    print(
        summary_output
    )


def main():

    print("\n")
    print("#" * 60)

    print(
        "TRAFFIC ACCIDENT HOTSPOT PREDICTION"
    )

    print(
        "SPATIAL GRID PROCESSING"
    )

    print("#" * 60)

    df = load_collision_data()

    df = create_spatial_grid(df)

    analyze_grid_distribution(df)

    grid_summary = create_grid_summary(
        df
    )

    save_grid_data(
        df,
        grid_summary
    )

    print_section(
        "SPATIAL GRID PROCESSING COMPLETE"
    )

    print(
        "Grid-based spatial dataset "
        "created successfully."
    )


if __name__ == "__main__":
    main()