"""Build the ML feature table (see features.py for the timing contract)."""

import pandas as pd

from config import PROCESSED_DATA_DIR, INTERIM_DATA_DIR, FEATURES_PATH
from features import build_feature_table, FEATURE_COLUMNS
from splits import assign_split

INPUT_PATH = PROCESSED_DATA_DIR / "grid_monthly_collisions.csv"


def main():
    monthly = pd.read_csv(INPUT_PATH, parse_dates=["year_month"])
    cluster_path = INTERIM_DATA_DIR / "grid_cluster_features.csv"
    clusters = pd.read_csv(cluster_path) if cluster_path.exists() else None
    if clusters is None:
        print("WARNING: grid_cluster_features.csv not found; run hotspot_detection.py first")
    df = build_feature_table(monthly, cluster_features=clusters)
    df["split"] = assign_split(df)
    print(f"Feature table: {df.shape}, features: {len(FEATURE_COLUMNS)}")
    print(df.groupby("split").agg(rows=("hotspot", "size"), positives=("hotspot", "sum"), rate=("hotspot", "mean")))
    df.to_parquet(FEATURES_PATH, index=False)
    print(f"Saved: {FEATURES_PATH}")


if __name__ == "__main__":
    main()
