"""DBSCAN collision hotspots, fit on the TRAINING period (2021-2023) only.

Pipeline role: clusters are mapped onto the 0.5 km grid and summarised as two
static per-grid features (grid_in_cluster, cluster_collision_density) that feed
the predictor. Because the fit uses training-period collisions only, no 2024+
information enters the cluster features.

Parameter choice (documented criterion, training data only):
  * an (eps, min_samples) pair is admissible if the largest cluster holds
    <= MAX_LARGEST_CLUSTER_SHARE of all clustered points (no chaining);
  * among admissible pairs pick the one clustering the most collisions
    (ties: fewer clusters).
The full sweep is written to results/dbscan_sweep.csv.
"""

import json
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from features import CLUSTER_FEATURES
from config import (
    INTERIM_DATA_DIR,
    RESULTS_DIR,
    TRAIN_END_MONTH,
    FIRST_MONTH,
)

EARTH_RADIUS_KM = 6371.0088
EPS_KM_CANDIDATES = (0.05, 0.10, 0.15, 0.20, 0.30)
MIN_SAMPLES_CANDIDATES = (10, 20, 40)
MAX_LARGEST_CLUSTER_SHARE = 0.05

GRID_INPUT_PATH = INTERIM_DATA_DIR / "collisions_with_grid.csv"
HOTSPOT_OUTPUT_PATH = INTERIM_DATA_DIR / "collisions_with_hotspots.csv"
HOTSPOT_SUMMARY_PATH = INTERIM_DATA_DIR / "hotspot_summary.csv"
CLUSTER_FEATURES_PATH = INTERIM_DATA_DIR / "grid_cluster_features.csv"



def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def training_period(df):
    """Collisions up to and including TRAIN_END_MONTH."""
    end = pd.Period(TRAIN_END_MONTH, "M").end_time
    return df[pd.to_datetime(df["datetime"]) <= end]


def fit_dbscan(df, eps_km, min_samples):
    """Cluster labels (-1 = noise) for `df` using haversine distance (lat, lon in radians)."""
    coords = np.radians(df[["latitude", "longitude"]].to_numpy())
    model = DBSCAN(
        eps=eps_km / EARTH_RADIUS_KM,
        min_samples=min_samples,
        metric="haversine",
        algorithm="ball_tree",
        n_jobs=-1,
    )
    return model.fit_predict(coords)


def cluster_stats(labels):
    clustered = labels[labels >= 0]
    sizes = np.bincount(clustered) if len(clustered) else np.array([0])
    return {
        "clusters": int((sizes > 0).sum()),
        "clustered_points": int(len(clustered)),
        "clustered_fraction": float(len(clustered) / len(labels)),
        "largest_cluster_points": int(sizes.max()),
        "largest_cluster_share": float(sizes.max() / max(len(clustered), 1)),
        "median_cluster_size": float(np.median(sizes[sizes > 0])) if len(clustered) else 0.0,
        "p95_cluster_size": float(np.percentile(sizes[sizes > 0], 95)) if len(clustered) else 0.0,
    }


def sweep_parameters(train):
    rows = []
    for eps in EPS_KM_CANDIDATES:
        for min_samples in MIN_SAMPLES_CANDIDATES:
            stats = cluster_stats(fit_dbscan(train, eps, min_samples))
            rows.append({"eps_km": eps, "min_samples": min_samples, **stats})
            print(f"eps={eps:.2f} km min_samples={min_samples}: {stats['clusters']:,} clusters, "
                  f"{stats['clustered_fraction']:.1%} clustered, largest share {stats['largest_cluster_share']:.1%}")
    return pd.DataFrame(rows)


def select_parameters(sweep):
    admissible = sweep[sweep["largest_cluster_share"] <= MAX_LARGEST_CLUSTER_SHARE]
    if admissible.empty:
        raise RuntimeError("no (eps, min_samples) pair avoids chaining; widen the candidate grid")
    best = admissible.sort_values(["clustered_fraction", "clusters"], ascending=[False, True]).iloc[0]
    return float(best["eps_km"]), int(best["min_samples"])


def grid_cluster_features(train_labeled, months_in_period):
    """Static per-grid cluster features from labelled TRAINING collisions.

    grid_in_cluster:          1 if >= 50% of the cell's training collisions are in a cluster.
    cluster_collision_density: collisions per cell-month of the cell's dominant cluster
                               (cluster collisions / (cells the cluster touches * months)).
    """
    df = train_labeled[["grid_id", "hotspot_id"]]
    total = df.groupby("grid_id").size().rename("total")
    clustered = df[df["hotspot_id"] >= 0]
    n_clustered = clustered.groupby("grid_id").size().rename("clustered")

    per_cluster = clustered.groupby("hotspot_id").agg(
        cluster_collisions=("grid_id", "size"), cluster_cells=("grid_id", "nunique")
    )
    per_cluster["cluster_collision_density"] = per_cluster["cluster_collisions"] / (
        per_cluster["cluster_cells"] * months_in_period
    )
    cell_cluster = clustered.groupby(["grid_id", "hotspot_id"]).size().rename("n").reset_index()
    dominant = cell_cluster.sort_values(["grid_id", "n", "hotspot_id"], ascending=[True, False, True]).drop_duplicates("grid_id")
    dominant = dominant.merge(per_cluster["cluster_collision_density"], left_on="hotspot_id", right_index=True)

    out = pd.concat([total, n_clustered], axis=1).fillna(0)
    out["grid_in_cluster"] = (out["clustered"] >= 0.5 * out["total"]).astype(int)
    out = out.join(dominant.set_index("grid_id")["cluster_collision_density"]).fillna({"cluster_collision_density": 0.0})
    return out[CLUSTER_FEATURES].reset_index()


def build_cluster_features(collisions, eps_km, min_samples):
    """End-to-end: restrict to the training period, cluster, map to grid cells.

    `collisions` needs: collision_index, latitude, longitude, datetime, grid_id.
    Anything after TRAIN_END_MONTH is ignored, by construction.
    """
    train = training_period(collisions).copy()
    train["hotspot_id"] = fit_dbscan(train, eps_km, min_samples)
    months = len(pd.period_range(FIRST_MONTH, TRAIN_END_MONTH, freq="M"))
    return train, grid_cluster_features(train, months)


def hotspot_summary(train_labeled):
    hot = train_labeled[train_labeled["hotspot_id"] >= 0]
    return (
        hot.groupby("hotspot_id")
        .agg(
            collision_count=("collision_index", "count"),
            average_latitude=("latitude", "mean"),
            average_longitude=("longitude", "mean"),
            grid_cells=("grid_id", "nunique"),
            total_casualties=("number_of_casualties", "sum"),
            first_collision=("datetime", "min"),
            last_collision=("datetime", "max"),
        )
        .reset_index()
        .sort_values("collision_count", ascending=False)
    )


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    print_section("LOADING COLLISIONS (TRAINING PERIOD ONLY FOR THE FIT)")
    df = pd.read_csv(GRID_INPUT_PATH, low_memory=False)
    df["datetime"] = pd.to_datetime(df["datetime"])
    train = training_period(df)
    print(f"All collisions: {len(df):,}   training-period collisions used: {len(train):,}")

    selection_path = RESULTS_DIR / "dbscan_selection.json"
    if "--reuse" in sys.argv and selection_path.exists():
        chosen = json.loads(selection_path.read_text())
        eps_km, min_samples = chosen["eps_km"], chosen["min_samples"]
        print(f"Reusing saved selection (training data unchanged): eps={eps_km} km, min_samples={min_samples}")
    else:
        print_section("PARAMETER SWEEP (2021-2023)")
        sweep = sweep_parameters(train)
        sweep.to_csv(RESULTS_DIR / "dbscan_sweep.csv", index=False)
        eps_km, min_samples = select_parameters(sweep)
        print(f"\nSelected: eps={eps_km} km, min_samples={min_samples}")

    print_section("FINAL FIT AND GRID MAPPING")
    train_labeled, features = build_cluster_features(df, eps_km, min_samples)
    stats = cluster_stats(train_labeled["hotspot_id"].to_numpy())
    print(json.dumps(stats, indent=2))

    train_labeled.to_csv(HOTSPOT_OUTPUT_PATH, index=False)
    hotspot_summary(train_labeled).to_csv(HOTSPOT_SUMMARY_PATH, index=False)
    features.to_csv(CLUSTER_FEATURES_PATH, index=False)
    (RESULTS_DIR / "dbscan_selection.json").write_text(json.dumps({
        "fit_period": f"{FIRST_MONTH}..{TRAIN_END_MONTH}",
        "criterion": f"largest cluster <= {MAX_LARGEST_CLUSTER_SHARE:.0%} of clustered points; then maximise clustered fraction",
        "eps_km": eps_km, "min_samples": min_samples, "stats": stats,
    }, indent=2))
    print(f"\nSaved: {HOTSPOT_OUTPUT_PATH.name}, {HOTSPOT_SUMMARY_PATH.name}, {CLUSTER_FEATURES_PATH.name}")


if __name__ == "__main__":
    main()
