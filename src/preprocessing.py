import pandas as pd

from config import (
    RAW_COLLISIONS_PATH,
    CLEAN_COLLISIONS_PATH,
    QUALITY_REPORT_PATH,
    REQUIRED_COLUMNS,
    UK_LAT_MIN,
    UK_LAT_MAX,
    UK_LON_MIN,
    UK_LON_MAX
)


def print_section(title):
    """Print a formatted section heading."""
    
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def load_data():
    """Load the raw collision dataset."""

    print_section("STEP 1: LOADING RAW DATA")

    df = pd.read_csv(
        RAW_COLLISIONS_PATH,
        low_memory=False
    )

    print(f"Raw dataset rows: {len(df):,}")
    print(f"Raw dataset columns: {len(df.columns)}")

    return df


def select_required_columns(df):
    """Select only columns required for the project."""

    print_section("STEP 2: SELECTING REQUIRED COLUMNS")

    available_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column in df.columns
    ]

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        print("WARNING: Missing expected columns:")
        print(missing_columns)

    df = df[available_columns].copy()

    print(f"Columns selected: {len(df.columns)}")

    print("\nSelected columns:")
    print(df.columns.tolist())

    return df


def remove_duplicates(df):
    """Remove duplicate collision records."""

    print_section("STEP 3: DUPLICATE CHECK")

    rows_before = len(df)

    duplicate_rows = df.duplicated().sum()

    print(f"Duplicate rows found: {duplicate_rows:,}")

    df = df.drop_duplicates()

    rows_after = len(df)

    print(f"Rows before duplicate removal: {rows_before:,}")
    print(f"Rows after duplicate removal: {rows_after:,}")

    return df


def check_missing_values(df):
    """Display missing values for all columns."""

    print_section("STEP 4: MISSING VALUE ANALYSIS")

    missing_values = df.isnull().sum()

    missing_report = pd.DataFrame({
        "column": missing_values.index,
        "missing_values": missing_values.values,
        "missing_percentage": (
            missing_values.values / len(df) * 100
        ).round(2)
    })

    missing_report = missing_report.sort_values(
        by="missing_values",
        ascending=False
    )

    print(missing_report.to_string(index=False))

    return df


def validate_coordinates(df):
    """Remove records with invalid latitude or longitude."""

    print_section("STEP 5: COORDINATE VALIDATION")

    rows_before = len(df)

    print(f"Latitude range allowed: {UK_LAT_MIN} to {UK_LAT_MAX}")
    print(f"Longitude range allowed: {UK_LON_MIN} to {UK_LON_MAX}")

    valid_coordinates = (

        df["latitude"].notna()
        &
        df["longitude"].notna()

        &
        df["latitude"].between(
            UK_LAT_MIN,
            UK_LAT_MAX
        )

        &
        df["longitude"].between(
            UK_LON_MIN,
            UK_LON_MAX
        )
    )

    invalid_coordinates = (~valid_coordinates).sum()

    print(
        f"Rows with invalid coordinates: "
        f"{invalid_coordinates:,}"
    )

    df = df[valid_coordinates].copy()

    rows_after = len(df)

    print(f"Rows before coordinate validation: {rows_before:,}")
    print(f"Rows after coordinate validation: {rows_after:,}")

    return df


def parse_datetime(df):
    """Parse date and time and create datetime column."""

    print_section("STEP 6: DATE AND TIME PARSING")

    print("Parsing date column...")

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
        dayfirst=True
    )

    print("Combining date and time...")

    datetime_string = (
        df["date"].dt.strftime("%Y-%m-%d")
        +
        " "
        +
        df["time"].astype(str)
    )

    df["datetime"] = pd.to_datetime(
        datetime_string,
        errors="coerce"
    )

    invalid_dates = df["date"].isna().sum()

    invalid_datetimes = df["datetime"].isna().sum()

    print(f"Invalid dates: {invalid_dates:,}")
    print(f"Invalid datetime values: {invalid_datetimes:,}")

    rows_before = len(df)

    df = df.dropna(
        subset=[
            "date",
            "datetime"
        ]
    )

    rows_after = len(df)

    print(f"Rows before datetime cleaning: {rows_before:,}")
    print(f"Rows after datetime cleaning: {rows_after:,}")

    return df


def create_quality_report(df):
    """Create a data quality report."""

    print_section("STEP 7: CREATING DATA QUALITY REPORT")

    report = pd.DataFrame({

        "column": df.columns,

        "data_type": [
            str(df[column].dtype)
            for column in df.columns
        ],

        "missing_values": [
            df[column].isna().sum()
            for column in df.columns
        ],

        "missing_percentage": [
            round(
                df[column].isna().mean() * 100,
                2
            )
            for column in df.columns
        ],

        "unique_values": [
            df[column].nunique()
            for column in df.columns
        ]
    })

    report.to_csv(
        QUALITY_REPORT_PATH,
        index=False
    )

    print(
        f"Quality report saved to:\n"
        f"{QUALITY_REPORT_PATH}"
    )

    return report


def save_clean_data(df):
    """Save cleaned dataset."""

    print_section("STEP 8: SAVING CLEAN DATA")

    df.to_csv(
        CLEAN_COLLISIONS_PATH,
        index=False
    )

    print(
        f"Clean dataset saved to:\n"
        f"{CLEAN_COLLISIONS_PATH}"
    )

    print(f"\nFinal rows: {len(df):,}")
    print(f"Final columns: {len(df.columns)}")


def run_preprocessing():
    """Run the complete preprocessing pipeline."""

    print("\n")
    print("#" * 60)
    print("TRAFFIC ACCIDENT HOTSPOT PREDICTION")
    print("DATA PREPROCESSING PIPELINE")
    print("#" * 60)

    df = load_data()

    raw_rows = len(df)

    df = select_required_columns(df)

    df = remove_duplicates(df)

    df = check_missing_values(df)

    df = validate_coordinates(df)

    df = parse_datetime(df)

    create_quality_report(df)

    save_clean_data(df)

    print_section("PREPROCESSING COMPLETE")

    print(f"Initial rows: {raw_rows:,}")
    print(f"Final rows: {len(df):,}")
    print(f"Rows removed: {raw_rows - len(df):,}")

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    run_preprocessing()