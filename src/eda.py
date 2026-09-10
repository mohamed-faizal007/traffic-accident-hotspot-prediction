import pandas as pd
import matplotlib.pyplot as plt

from config import (
    CLEAN_COLLISIONS_PATH,
    EDA_FIGURES_DIR
)


def print_section(title):
    """Print a formatted section heading."""

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def load_clean_data():
    """Load the cleaned collision dataset."""

    print_section("STEP 1: LOADING CLEAN DATA")

    df = pd.read_csv(
        CLEAN_COLLISIONS_PATH,
        low_memory=False
    )

    df["date"] = pd.to_datetime(df["date"])
    df["datetime"] = pd.to_datetime(df["datetime"])

    print("Dataset loaded successfully")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    return df


def dataset_overview(df):
    """Display basic dataset information."""

    print_section("STEP 2: DATASET OVERVIEW")

    print(f"Dataset Shape: {df.shape}")

    print("\nDate Range:")
    print("Start:", df["datetime"].min())
    print("End:", df["datetime"].max())

    print("\nUnique Years:")
    print(
        sorted(
            df["collision_year"].unique()
        )
    )


def save_bar_chart(
    counts,
    title,
    xlabel,
    filename,
    rotate_labels=False
):
    """Create and save a bar chart."""

    plt.figure(figsize=(12, 6))

    plt.bar(
        counts.index.astype(str),
        counts.values
    )

    plt.title(title)

    plt.xlabel(xlabel)

    plt.ylabel(
        "Number of Collisions"
    )

    if rotate_labels:
        plt.xticks(rotation=45)

    plt.tight_layout()

    output_path = (
        EDA_FIGURES_DIR
        / filename
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"\nFigure saved:\n{output_path}"
    )


def analyze_collisions_by_year(df):
    """Analyze collisions by year."""

    print_section(
        "STEP 3: COLLISIONS BY YEAR"
    )

    yearly_counts = (
        df["collision_year"]
        .value_counts()
        .sort_index()
    )

    print(yearly_counts)

    save_bar_chart(
        yearly_counts,
        "Traffic Collisions by Year",
        "Year",
        "collisions_by_year.png"
    )


def analyze_collisions_by_month(df):
    """Analyze collision patterns by month."""

    print_section(
        "STEP 4: COLLISIONS BY MONTH"
    )

    month_counts = (
        df["datetime"]
        .dt.month
        .value_counts()
        .sort_index()
    )

    print(month_counts)

    plt.figure(figsize=(10, 6))

    plt.plot(
        month_counts.index,
        month_counts.values,
        marker="o"
    )

    plt.title(
        "Traffic Collisions by Month"
    )

    plt.xlabel("Month")

    plt.ylabel(
        "Number of Collisions"
    )

    plt.xticks(range(1, 13))

    plt.tight_layout()

    output_path = (
        EDA_FIGURES_DIR
        / "collisions_by_month.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"\nFigure saved:\n{output_path}"
    )


def analyze_collisions_by_day(df):
    """Analyze collisions by day of week."""

    print_section(
        "STEP 5: COLLISIONS BY DAY OF WEEK"
    )

    daily_counts = (
        df["day_of_week"]
        .value_counts()
        .sort_index()
    )

    print(daily_counts)

    save_bar_chart(
        daily_counts,
        "Traffic Collisions by Day of Week",
        "Day of Week Code",
        "collisions_by_day.png"
    )


def analyze_collisions_by_hour(df):
    """Analyze collision patterns by hour."""

    print_section(
        "STEP 6: COLLISIONS BY HOUR"
    )

    hourly_counts = (
        df["datetime"]
        .dt.hour
        .value_counts()
        .sort_index()
    )

    print(hourly_counts)

    plt.figure(figsize=(12, 6))

    plt.plot(
        hourly_counts.index,
        hourly_counts.values,
        marker="o"
    )

    plt.title(
        "Traffic Collisions by Hour"
    )

    plt.xlabel("Hour of Day")

    plt.ylabel(
        "Number of Collisions"
    )

    plt.xticks(range(24))

    plt.tight_layout()

    output_path = (
        EDA_FIGURES_DIR
        / "collisions_by_hour.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"\nFigure saved:\n{output_path}"
    )


def analyze_collision_severity(df):
    """Analyze collision severity."""

    print_section(
        "STEP 7: COLLISION SEVERITY ANALYSIS"
    )

    severity_counts = (
        df["collision_severity"]
        .value_counts()
        .sort_index()
    )

    print(
        "Collision Severity Distribution:"
    )

    print(severity_counts)

    print(
        "\nCollision Severity Percentage:"
    )

    severity_percentage = (
        df["collision_severity"]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    ).round(2)

    print(severity_percentage)

    save_bar_chart(
        severity_counts,
        "Collision Severity Distribution",
        "Severity Code",
        "collision_severity.png"
    )


def analyze_categorical_feature(
    df,
    column,
    step_number,
    title,
    filename
):
    """Analyze a categorical collision feature."""

    print_section(
        f"STEP {step_number}: {title.upper()}"
    )

    counts = (
        df[column]
        .value_counts()
        .sort_index()
    )

    print(counts)

    print(
        "\nPercentage Distribution:"
    )

    percentages = (
        df[column]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    ).round(2)

    print(percentages)

    save_bar_chart(
        counts,
        title,
        column.replace(
            "_",
            " "
        ).title(),
        filename,
        rotate_labels=True
    )


def analyze_speed_limit(df):
    """Analyze speed limit distribution."""

    print_section(
        "STEP 12: SPEED LIMIT ANALYSIS"
    )

    speed_counts = (
        df["speed_limit"]
        .value_counts()
        .sort_index()
    )

    print(speed_counts)

    save_bar_chart(
        speed_counts,
        "Traffic Collisions by Speed Limit",
        "Speed Limit",
        "collisions_by_speed_limit.png"
    )


def analyze_spatial_distribution(df):
    """Analyze geographic distribution of collisions."""

    print_section(
        "STEP 14: SPATIAL DISTRIBUTION ANALYSIS"
    )

    print("Latitude Range:")

    print(
        "Minimum:",
        df["latitude"].min()
    )

    print(
        "Maximum:",
        df["latitude"].max()
    )

    print("\nLongitude Range:")

    print(
        "Minimum:",
        df["longitude"].min()
    )

    print(
        "Maximum:",
        df["longitude"].max()
    )

    print("\nUnique Coordinates:")

    unique_coordinates = (
        df[
            [
                "latitude",
                "longitude"
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    print(
        f"{unique_coordinates:,}"
    )

    print(
        "\nCreating spatial scatter plot..."
    )

    plt.figure(figsize=(10, 12))

    plt.scatter(
        df["longitude"],
        df["latitude"],
        s=1,
        alpha=0.1
    )

    plt.title(
        "Geographic Distribution of Traffic Collisions"
    )

    plt.xlabel("Longitude")

    plt.ylabel("Latitude")

    plt.tight_layout()

    output_path = (
        EDA_FIGURES_DIR
        / "spatial_distribution.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"\nFigure saved:\n{output_path}"
    )


def main():

    print("\n")
    print("#" * 60)
    print("TRAFFIC ACCIDENT HOTSPOT PREDICTION")
    print("EXPLORATORY DATA ANALYSIS")
    print("#" * 60)

    EDA_FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = load_clean_data()

    dataset_overview(df)

    analyze_collisions_by_year(df)

    analyze_collisions_by_month(df)

    analyze_collisions_by_day(df)

    analyze_collisions_by_hour(df)

    analyze_collision_severity(df)

    analyze_categorical_feature(
        df,
        "weather_conditions",
        8,
        "Weather Conditions",
        "weather_conditions.png"
    )

    analyze_categorical_feature(
        df,
        "road_surface_conditions",
        9,
        "Road Surface Conditions",
        "road_surface_conditions.png"
    )

    analyze_categorical_feature(
        df,
        "light_conditions",
        10,
        "Light Conditions",
        "light_conditions.png"
    )

    analyze_categorical_feature(
        df,
        "road_type",
        11,
        "Road Type",
        "road_type.png"
    )

    analyze_speed_limit(df)

    analyze_categorical_feature(
        df,
        "urban_or_rural_area",
        13,
        "Urban vs Rural Area",
        "urban_vs_rural.png"
    )

    analyze_spatial_distribution(df)

    print_section(
        "EDA STEP COMPLETE"
    )

    print(
        "Complete exploratory data analysis "
        "completed successfully."
    )


if __name__ == "__main__":
    main()