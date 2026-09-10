import pandas as pd

from config import PROCESSED_DATA_DIR


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DATA_PATH = (
    PROCESSED_DATA_DIR
    / "ml_features.csv"
)

OUTPUT_DATA_PATH = (
    PROCESSED_DATA_DIR
    / "ml_features_advanced.csv"
)


# ============================================================
# HELPER FUNCTION
# ============================================================

def print_section(title):
    """Print formatted section heading."""

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ============================================================
# STEP 1: LOAD DATASET
# ============================================================

def load_dataset():
    """Load existing machine learning dataset."""

    print_section(
        "STEP 1: LOADING MACHINE LEARNING DATASET"
    )

    df = pd.read_csv(
        INPUT_DATA_PATH,
        low_memory=False
    )

    print(
        f"Total records: {len(df):,}"
    )

    print(
        f"Total columns: {len(df.columns)}"
    )

    print(
        f"Unique grids: "
        f"{df['grid_id'].nunique():,}"
    )

    print(
        f"\nDataset loaded from:"
        f"\n{INPUT_DATA_PATH}"
    )

    return df


# ============================================================
# STEP 2: PREPARE TEMPORAL ORDER
# ============================================================

def prepare_temporal_order(df):
    """
    Sort dataset by spatial grid and time.

    This is required before creating
    historical features.
    """

    print_section(
        "STEP 2: PREPARING TEMPORAL ORDER"
    )

    df["year_month"] = pd.to_datetime(
        df["year_month"]
    )

    df = df.sort_values(
        by=[
            "grid_id",
            "year_month"
        ]
    ).reset_index(
        drop=True
    )

    print(
        "Dataset sorted by:"
    )

    print(
        "- grid_id"
    )

    print(
        "- year_month"
    )

    print(
        "\nTemporal ordering completed."
    )

    return df


# ============================================================
# STEP 3: HISTORICAL COLLISION FEATURES
# ============================================================

def create_historical_collision_features(df):
    """
    Create historical collision statistics.

    All features use only information
    from previous months.

    shift(1) prevents temporal leakage.
    """

    print_section(
        "STEP 3: CREATING HISTORICAL COLLISION FEATURES"
    )

    grouped = df.groupby(
        "grid_id"
    )["collision_count"]


    # --------------------------------------------------------
    # Historical Total Collisions
    # --------------------------------------------------------

    df[
        "historical_total_collisions"
    ] = (
        grouped
        .cumsum()
        - df["collision_count"]
    )


    # --------------------------------------------------------
    # Historical Average Collisions
    # --------------------------------------------------------

    df[
        "historical_avg_collisions"
    ] = (
        grouped
        .transform(
            lambda x:
            x.expanding()
            .mean()
            .shift(1)
        )
    )


    # --------------------------------------------------------
    # Historical Maximum Collisions
    # --------------------------------------------------------

    df[
        "historical_max_collisions"
    ] = (
        grouped
        .transform(
            lambda x:
            x.expanding()
            .max()
            .shift(1)
        )
    )


    # --------------------------------------------------------
    # Historical Collision Frequency
    # --------------------------------------------------------

    df[
        "historical_collision_frequency"
    ] = (
        grouped
        .transform(
            lambda x:
            x.gt(0)
            .expanding()
            .mean()
            .shift(1)
        )
    )


    print(
        "Created features:"
    )

    print(
        "- historical_total_collisions"
    )

    print(
        "- historical_avg_collisions"
    )

    print(
        "- historical_max_collisions"
    )

    print(
        "- historical_collision_frequency"
    )

    return df


# ============================================================
# STEP 4: HISTORICAL HOTSPOT FEATURES
# ============================================================

def create_historical_hotspot_features(df):
    """
    Create historical hotspot features.

    Uses previous collision information
    only.
    """

    print_section(
        "STEP 4: CREATING HISTORICAL HOTSPOT FEATURES"
    )


    # --------------------------------------------------------
    # Historical Hotspot Indicator
    # --------------------------------------------------------

    historical_hotspot = (
        df["collision_count"] >= 2
    ).astype(int)


    # --------------------------------------------------------
    # Historical Hotspot Frequency
    # --------------------------------------------------------

    df[
        "historical_hotspot_frequency"
    ] = (
        historical_hotspot
        .groupby(
            df["grid_id"]
        )
        .transform(
            lambda x:
            x.expanding()
            .mean()
            .shift(1)
        )
    )


    print(
        "Created features:"
    )

    print(
        "- historical_hotspot_frequency"
    )

    return df


# ============================================================
# STEP 5: RECENT COLLISION ACTIVITY
# ============================================================

def create_recent_activity_features(df):
    """
    Create recent collision activity features.

    These features capture short-term
    temporal accident patterns.
    """

    print_section(
        "STEP 5: CREATING RECENT ACTIVITY FEATURES"
    )

    grouped = df.groupby(
        "grid_id"
    )["collision_count"]


    # --------------------------------------------------------
    # Previous 3 Month Collision Sum
    # --------------------------------------------------------

    df[
        "collisions_last_3_months"
    ] = (
        grouped
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


    # --------------------------------------------------------
    # Previous 6 Month Collision Sum
    # --------------------------------------------------------

    df[
        "collisions_last_6_months"
    ] = (
        grouped
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                window=6,
                min_periods=1
            )
            .sum()
        )
    )


    # --------------------------------------------------------
    # Previous 12 Month Collision Sum
    # --------------------------------------------------------

    df[
        "collisions_last_12_months"
    ] = (
        grouped
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                window=12,
                min_periods=1
            )
            .sum()
        )
    )


    print(
        "Created features:"
    )

    print(
        "- collisions_last_3_months"
    )

    print(
        "- collisions_last_6_months"
    )

    print(
        "- collisions_last_12_months"
    )

    return df


# ============================================================
# STEP 6: COLLISION RECENCY FEATURE
# ============================================================

def create_collision_recency_feature(df):
    """
    Calculate months since the previous
    collision for each grid.

    Uses only historical information.
    """

    print_section(
        "STEP 6: CREATING COLLISION RECENCY FEATURE"
    )


    # Month number for calculations

    df["_month_index"] = (
        df["year"] * 12
        + df["month"]
    )


    # Store month index only
    # when collision occurred

    collision_month = (
        df["_month_index"]
        .where(
            df["collision_count"] > 0
        )
    )


    # Previous collision month

    previous_collision_month = (
        collision_month
        .groupby(
            df["grid_id"]
        )
        .ffill()
        .groupby(
            df["grid_id"]
        )
        .shift(1)
    )


    # Months since previous collision

    df[
        "months_since_last_collision"
    ] = (
        df["_month_index"]
        - previous_collision_month
    )


    # Remove temporary column

    df.drop(
        columns=[
            "_month_index"
        ],
        inplace=True
    )


    print(
        "Created feature:"
    )

    print(
        "- months_since_last_collision"
    )

    return df


# ============================================================
# STEP 7: HANDLE MISSING VALUES
# ============================================================

def handle_missing_values(df):
    """
    Fill missing values created by
    historical calculations.
    """

    print_section(
        "STEP 7: HANDLING MISSING VALUES"
    )


    advanced_features = [

        "historical_total_collisions",

        "historical_avg_collisions",

        "historical_max_collisions",

        "historical_collision_frequency",

        "historical_hotspot_frequency",

        "collisions_last_3_months",

        "collisions_last_6_months",

        "collisions_last_12_months",

        "months_since_last_collision"
    ]


    print(
        "Missing values before handling:"
    )

    print(
        df[
            advanced_features
        ].isnull().sum()
    )


    # Fill historical features

    df[
        advanced_features
    ] = df[
        advanced_features
    ].fillna(0)


    print(
        "\nMissing values after handling:"
    )

    print(
        df[
            advanced_features
        ].isnull().sum()
    )


    print(
        "\nMissing values handled successfully."
    )

    return df


# ============================================================
# STEP 8: ANALYZE FEATURES
# ============================================================

def analyze_advanced_features(df):
    """Display advanced feature statistics."""

    print_section(
        "STEP 8: ANALYZING ADVANCED FEATURES"
    )


    advanced_features = [

        "historical_total_collisions",

        "historical_avg_collisions",

        "historical_max_collisions",

        "historical_collision_frequency",

        "historical_hotspot_frequency",

        "collisions_last_3_months",

        "collisions_last_6_months",

        "collisions_last_12_months",

        "months_since_last_collision"
    ]


    print(
        "Advanced Feature Statistics:\n"
    )


    print(
        df[
            advanced_features
        ]
        .describe()
        .round(3)
    )


# ============================================================
# STEP 9: SAVE DATASET
# ============================================================

def save_dataset(df):
    """Save advanced machine learning dataset."""

    print_section(
        "STEP 9: SAVING ADVANCED DATASET"
    )


    df.to_csv(
        OUTPUT_DATA_PATH,
        index=False
    )


    print(
        "Advanced dataset saved successfully:"
    )

    print(
        OUTPUT_DATA_PATH
    )


    print(
        f"\nFinal records: "
        f"{len(df):,}"
    )


    print(
        f"Final columns: "
        f"{len(df.columns)}"
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("\n")

    print("#" * 60)

    print(
        "TRAFFIC ACCIDENT HOTSPOT PREDICTION"
    )

    print(
        "ADVANCED FEATURE ENGINEERING"
    )

    print("#" * 60)


    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    df = load_dataset()


    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    df = prepare_temporal_order(
        df
    )


    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    df = (
        create_historical_collision_features(
            df
        )
    )


    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    df = (
        create_historical_hotspot_features(
            df
        )
    )


    # --------------------------------------------------------
    # STEP 5
    # --------------------------------------------------------

    df = (
        create_recent_activity_features(
            df
        )
    )


    # --------------------------------------------------------
    # STEP 6
    # --------------------------------------------------------

    df = (
        create_collision_recency_feature(
            df
        )
    )


    # --------------------------------------------------------
    # STEP 7
    # --------------------------------------------------------

    df = handle_missing_values(
        df
    )


    # --------------------------------------------------------
    # STEP 8
    # --------------------------------------------------------

    analyze_advanced_features(
        df
    )


    # --------------------------------------------------------
    # STEP 9
    # --------------------------------------------------------

    save_dataset(
        df
    )


    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print_section(
        "ADVANCED FEATURE ENGINEERING COMPLETE"
    )


    print(
        "Advanced spatio-temporal features "
        "created successfully."
    )


    print(
        "\nNew features added:"
    )


    print(
        "- historical_total_collisions"
    )

    print(
        "- historical_avg_collisions"
    )

    print(
        "- historical_max_collisions"
    )

    print(
        "- historical_collision_frequency"
    )

    print(
        "- historical_hotspot_frequency"
    )

    print(
        "- collisions_last_3_months"
    )

    print(
        "- collisions_last_6_months"
    )

    print(
        "- collisions_last_12_months"
    )

    print(
        "- months_since_last_collision"
    )


if __name__ == "__main__":

    main()