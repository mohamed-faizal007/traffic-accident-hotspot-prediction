"""Cluster table for the dashboard from the saved DBSCAN fit (2021-2023 collisions, eps 0.1 km, min 10).

Read-only over data/interim/hotspot_summary.csv and collisions_with_hotspots.csv. Before exporting it
checks that the saved fit is the one recorded in results/dbscan_selection.json (cluster count, largest
cluster, clustered points) and aborts otherwise. Per cluster: centroid, size, extent (bounding box and
max distance from the centroid), grid cells, casualties, first/last collision.
Writes only outputs/dashboard/dbscan_clusters.csv.
"""

import json

import numpy as np
import pandas as pd

try:
    from ._common import DASHBOARD_DIR, PROJECT_ROOT
except ImportError:  # run as a script
    from _common import DASHBOARD_DIR, PROJECT_ROOT

SUMMARY = PROJECT_ROOT / "data" / "interim" / "hotspot_summary.csv"
POINTS = PROJECT_ROOT / "data" / "interim" / "collisions_with_hotspots.csv"
SELECTION = PROJECT_ROOT / "results" / "dbscan_selection.json"
OUT = DASHBOARD_DIR / "dbscan_clusters.csv"
EARTH_RADIUS_KM = 6371.0088


def main():
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    sel = json.loads(SELECTION.read_text())
    stats = sel["stats"]
    summary = pd.read_csv(SUMMARY)
    if not (len(summary) == stats["clusters"] and summary["collision_count"].max() == stats["largest_cluster_points"]
            and summary["collision_count"].sum() == stats["clustered_points"]):
        raise SystemExit("hotspot_summary.csv is not the fit recorded in results/dbscan_selection.json")

    pts = pd.read_csv(POINTS, usecols=["hotspot_id", "latitude", "longitude"])
    pts = pts[pts["hotspot_id"] >= 0]
    g = pts.groupby("hotspot_id")
    ext = g.agg(lat_min=("latitude", "min"), lat_max=("latitude", "max"),
                lon_min=("longitude", "min"), lon_max=("longitude", "max"))
    pts = pts.join(summary.set_index("hotspot_id")[["average_latitude", "average_longitude"]], on="hotspot_id")
    la1, lo1 = np.radians(pts["latitude"]), np.radians(pts["longitude"])
    la2, lo2 = np.radians(pts["average_latitude"]), np.radians(pts["average_longitude"])
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    pts["d_km"] = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))
    ext["radius_km"] = pts.groupby("hotspot_id")["d_km"].max()

    out = summary.merge(ext, left_on="hotspot_id", right_index=True, validate="one_to_one")
    out = out.rename(columns={"collision_count": "size", "average_latitude": "lat", "average_longitude": "lon"})
    out = out.sort_values("size", ascending=False)
    out["first_collision"] = out["first_collision"].str[:10]
    out["last_collision"] = out["last_collision"].str[:10]
    out.to_csv(OUT, index=False, float_format="%.5f")
    print(f"{len(out):,} clusters (largest {out['size'].max():,}) -> {OUT}")


if __name__ == "__main__":
    main()
