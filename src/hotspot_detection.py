import pandas as pd
import numpy as np

from sklearn.cluster import DBSCAN

from config import (
    CLEAN_COLLISIONS_PATH
)


EARTH_RADIUS_KM = 6371.0088

HOTSPOT_RADIUS_KM = 0.3

MIN_SAMPLES = 10

HOTSPOT_OUTPUT_PATH = (
    "data/interim/collisions_with_hotspots.csv"
)

HOTSPOT_SUMMARY_PATH = (
    "data/interim/hotspot_summary.csv"
)


def print_section(title):
    """Print a formatted section heading."""

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def load_spatial_data():
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


def prepare_coordinates(df):
    """Convert coordinates to radians."""

    print_section(
        "STEP 2: PREPARING COORDINATES"
    )

    coordinates = df[
        [
            "latitude",
            "longitude"
        ]
    ].to_numpy()

    coordinates_radians = np.radians(
        coordinates
    )

    print(
        f"Coordinates prepared: "
        f"{len(coordinates_radians):,}"
    )

    print(
        "Coordinates converted to radians."
    )

    return coordinates_radians


def detect_hotspots(coordinates_radians):
    """Run final DBSCAN hotspot detection."""

    print_section(
        "STEP 3: RUNNING FINAL DBSCAN"
    )

    eps = (
        HOTSPOT_RADIUS_KM
        / EARTH_RADIUS_KM
    )

    print(
        f"Hotspot radius: "
        f"{HOTSPOT_RADIUS_KM} km"
    )

    print(
        f"Minimum samples: "
        f"{MIN_SAMPLES}"
    )

    print(
        f"DBSCAN eps: {eps}"
    )

    print(
        "\nRunning DBSCAN..."
    )

    dbscan = DBSCAN(
        eps=eps,
        min_samples=MIN_SAMPLES,
        metric="haversine",
        algorithm="ball_tree",
        n_jobs=-1
    )

    labels = dbscan.fit_predict(
        coordinates_radians
    )

    print(
        "DBSCAN completed successfully."
    )

    return labels


def assign_hotspot_labels(df, labels):
    """Assign DBSCAN cluster labels."""

    print_section(
        "STEP 4: ASSIGNING HOTSPOT LABELS"
    )

    df["hotspot_id"] = labels

    df["is_hotspot"] = (
        df["hotspot_id"] != -1
    )

    hotspot_points = (
        df["is_hotspot"].sum()
    )

    noise_points = (
        (~df["is_hotspot"]).sum()
    )

    print(
        f"Hotspot collisions: "
        f"{hotspot_points:,}"
    )

    print(
        f"Noise collisions: "
        f"{noise_points:,}"
    )

    return df


def create_hotspot_summary(df):
    """Create statistics for each hotspot."""

    print_section(
        "STEP 5: CREATING HOTSPOT SUMMARY"
    )

    hotspot_df = df[
        df["is_hotspot"]
    ].copy()

    summary = (
        hotspot_df
        .groupby("hotspot_id")
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

    summary = summary.sort_values(
        "collision_count",
        ascending=False
    )

    print(
        f"Total hotspots created: "
        f"{len(summary):,}"
    )

    print(
        "\nTop 10 hotspots:"
    )

    print(
        summary.head(10).to_string(
            index=False
        )
    )

    return summary


def save_hotspot_data(df, summary):
    """Save hotspot datasets."""

    print_section(
        "STEP 6: SAVING HOTSPOT DATA"
    )

    df.to_csv(
        HOTSPOT_OUTPUT_PATH,
        index=False
    )

    summary.to_csv(
        HOTSPOT_SUMMARY_PATH,
        index=False
    )

    print(
        "Collision hotspot dataset saved:"
    )

    print(
        HOTSPOT_OUTPUT_PATH
    )

    print(
        "\nHotspot summary saved:"
    )

    print(
        HOTSPOT_SUMMARY_PATH
    )


def main():

    print("\n")
    print("#" * 60)

    print(
        "TRAFFIC ACCIDENT HOTSPOT PREDICTION"
    )

    print(
        "FINAL SPATIAL HOTSPOT DETECTION"
    )

    print("#" * 60)

    df = load_spatial_data()

    coordinates_radians = (
        prepare_coordinates(df)
    )

    labels = detect_hotspots(
        coordinates_radians
    )

    df = assign_hotspot_labels(
        df,
        labels
    )

    summary = create_hotspot_summary(
        df
    )

    save_hotspot_data(
        df,
        summary
    )

    print_section(
        "HOTSPOT DETECTION COMPLETE"
    )

    print(
        "Final spatial hotspot dataset "
        "created successfully."
    )


if __name__ == "__main__":
    main()