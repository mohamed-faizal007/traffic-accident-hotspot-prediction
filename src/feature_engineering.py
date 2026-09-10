import pandas as pd
import numpy as np
from pathlib import Path


INPUT_PATH = Path(
    "data/processed/grid_monthly_collisions.csv"
)

OUTPUT_PATH = Path(
    "data/processed/ml_features.csv"
)


HOTSPOT_THRESHOLD = 2


def print_section(title):
    """Print a formatted section heading."""

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def load_temporal_dataset():
    """Load the grid-based temporal dataset."""

    print_section(
        "STEP 1: LOADING TEMPORAL DATASET"
    )

    df = pd.read_csv(
        INPUT_PATH,
        low_memory=False
    )

    df["year_month"] = pd.to_datetime(
        df["year_month"]
    )

    print(
        f"Total records: {len(df):,}"
    )

    print(
        f"Unique grids: "
        f"{df['grid_id'].nunique():,}"
    )

    print(
        f"Date range: "
        f"{df['year_month'].min().date()} "
        f"to "
        f"{df['year_month'].max().date()}"
    )

    return df


def sort_temporal_data(df):
    """Sort records for time-series feature creation."""

    print_section(
        "STEP 2: SORTING TEMPORAL DATA"
    )

    df = df.sort_values(
        [
            "grid_id",
            "year_month"
        ]
    ).reset_index(
        drop=True
    )

    print(
        "Data sorted by grid and month."
    )

    return df


def create_lag_features(df):
    """Create historical accident lag features."""

    print_section(
        "STEP 3: CREATING LAG FEATURES"
    )

    df["collisions_lag_1"] = (
        df.groupby("grid_id")[
            "collision_count"
        ].shift(1)
    )

    df["collisions_lag_2"] = (
        df.groupby("grid_id")[
            "collision_count"
        ].shift(2)
    )

    df["collisions_lag_3"] = (
        df.groupby("grid_id")[
            "collision_count"
        ].shift(3)
    )

    df["casualties_lag_1"] = (
        df.groupby("grid_id")[
            "total_casualties"
        ].shift(1)
    )

    print(
        "Created features:"
    )

    print("- collisions_lag_1")
    print("- collisions_lag_2")
    print("- collisions_lag_3")
    print("- casualties_lag_1")

    return df


def create_rolling_features(df):
    """Create rolling historical statistics."""

    print_section(
        "STEP 4: CREATING ROLLING FEATURES"
    )

    df["collisions_rolling_3"] = (
        df.groupby("grid_id")[
            "collision_count"
        ]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                window=3,
                min_periods=1
            )
            .mean()
        )
    )

    df["collisions_rolling_6"] = (
        df.groupby("grid_id")[
            "collision_count"
        ]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                window=6,
                min_periods=1
            )
            .mean()
        )
    )

    df["collisions_rolling_sum_3"] = (
        df.groupby("grid_id")[
            "collision_count"
        ]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                window=3,
                min_periods=1
            )
            .sum()
        )
    )

    print(
        "Created rolling features:"
    )

    print("- collisions_rolling_3")
    print("- collisions_rolling_6")
    print("- collisions_rolling_sum_3")

    return df


def create_seasonal_features(df):
    """Create cyclical month features."""

    print_section(
        "STEP 5: CREATING SEASONAL FEATURES"
    )

    df["month_sin"] = np.sin(
        2
        * np.pi
        * df["month"]
        / 12
    )

    df["month_cos"] = np.cos(
        2
        * np.pi
        * df["month"]
        / 12
    )

    print(
        "Created features:"
    )

    print("- month_sin")
    print("- month_cos")

    return df


def create_prediction_target(df):
    """
    Create next-month collision target
    and hotspot classification label.
    """

    print_section(
        "STEP 6: CREATING PREDICTION TARGET"
    )

    df["next_month_collisions"] = (
        df.groupby("grid_id")[
            "collision_count"
        ].shift(-1)
    )

    df["hotspot"] = (
        df["next_month_collisions"]
        >= HOTSPOT_THRESHOLD
    ).astype(int)

    print(
        f"Hotspot threshold: "
        f"{HOTSPOT_THRESHOLD} "
        f"collisions next month"
    )

    hotspot_count = (
        df["hotspot"] == 1
    ).sum()

    non_hotspot_count = (
        df["hotspot"] == 0
    ).sum()

    print(
        f"Hotspot records: "
        f"{hotspot_count:,}"
    )

    print(
        f"Non-hotspot records: "
        f"{non_hotspot_count:,}"
    )

    print(
        f"Hotspot percentage: "
        f"{hotspot_count / len(df) * 100:.2f}%"
    )

    return df


def remove_invalid_records(df):
    """
    Remove records without enough
    historical or future information.
    """

    print_section(
        "STEP 7: REMOVING INVALID RECORDS"
    )

    initial_rows = len(df)

    df = df.dropna(
        subset=[
            "collisions_lag_1",
            "collisions_lag_2",
            "collisions_lag_3",
            "next_month_collisions"
        ]
    )

    final_rows = len(df)

    print(
        f"Initial records: "
        f"{initial_rows:,}"
    )

    print(
        f"Final ML records: "
        f"{final_rows:,}"
    )

    print(
        f"Records removed: "
        f"{initial_rows - final_rows:,}"
    )

    return df


def analyze_ml_dataset(df):
    """Analyze final ML dataset."""

    print_section(
        "STEP 8: ANALYZING ML DATASET"
    )

    print(
        f"Final dataset shape: "
        f"{df.shape}"
    )

    print(
        "\nHotspot distribution:"
    )

    print(
        df["hotspot"]
        .value_counts()
    )

    print(
        "\nHotspot percentage:"
    )

    print(
        (
            df["hotspot"]
            .value_counts(
                normalize=True
            )
            * 100
        )
        .round(2)
    )

    print(
        "\nFeature columns:"
    )

    for column in df.columns:

        print(
            f"- {column}"
        )


def save_ml_dataset(df):
    """Save machine learning dataset."""

    print_section(
        "STEP 9: SAVING ML DATASET"
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
        "ML dataset saved:"
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
        "FEATURE ENGINEERING"
    )

    print("#" * 60)

    df = load_temporal_dataset()

    df = sort_temporal_data(df)

    df = create_lag_features(df)

    df = create_rolling_features(df)

    df = create_seasonal_features(df)

    df = create_prediction_target(df)

    df = remove_invalid_records(df)

    analyze_ml_dataset(df)

    save_ml_dataset(df)

    print_section(
        "FEATURE ENGINEERING COMPLETE"
    )

    print(
        "Machine learning dataset "
        "created successfully."
    )


if __name__ == "__main__":
    main()