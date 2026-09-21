"""No 2024+ information may enter the DBSCAN fit or the cluster features."""

import numpy as np
import pandas as pd

from features import CLUSTER_FEATURES, build_feature_table
from hotspot_detection import build_cluster_features, cluster_stats, fit_dbscan

EPS_KM, MIN_SAMPLES = 0.1, 10


def blob(rng, lat, lon, n, date, grid_id, start_index):
    """n collisions within ~30 m of (lat, lon)."""
    return pd.DataFrame({
        "collision_index": [f"c{start_index + i}" for i in range(n)],
        "latitude": lat + rng.normal(0, 0.0002, n),
        "longitude": lon + rng.normal(0, 0.0002, n),
        "datetime": pd.to_datetime(date) + pd.to_timedelta(rng.integers(0, 300, n), unit="D"),
        "grid_id": grid_id,
        "number_of_casualties": 1,
    })


def make_collisions():
    rng = np.random.default_rng(0)
    early = blob(rng, 51.5, -0.1, 40, "2022-01-01", "G_early", 0)      # training-period blob
    late = blob(rng, 52.5, -1.5, 40, "2024-03-01", "G_late", 100)      # dense blob, ONLY 2024
    sparse = blob(rng, 53.5, -2.5, 3, "2022-06-01", "G_sparse", 200)   # too few points
    return pd.concat([early, late, sparse], ignore_index=True)


def test_post_training_collisions_do_not_form_clusters():
    train, feats = build_cluster_features(make_collisions(), EPS_KM, MIN_SAMPLES)
    assert train["datetime"].max() <= pd.Timestamp("2023-12-31 23:59:59")
    f = feats.set_index("grid_id")
    assert f.loc["G_early", "grid_in_cluster"] == 1 and f.loc["G_early", "cluster_collision_density"] > 0
    assert "G_late" not in f.index                       # a 2024-only blob is invisible
    assert f.loc["G_sparse", "grid_in_cluster"] == 0


def test_cluster_features_unchanged_by_future_data():
    base = make_collisions()
    _, f1 = build_cluster_features(base, EPS_KM, MIN_SAMPLES)
    rng = np.random.default_rng(1)
    future = pd.concat([  # add lots of 2024/2025 collisions, incl. on top of the early cluster
        blob(rng, 51.5, -0.1, 200, "2024-06-01", "G_early", 1000),
        blob(rng, 53.5, -2.5, 200, "2025-02-01", "G_sparse", 2000),
    ])
    _, f2 = build_cluster_features(pd.concat([base, future], ignore_index=True), EPS_KM, MIN_SAMPLES)
    pd.testing.assert_frame_equal(f1.sort_values("grid_id").reset_index(drop=True),
                                  f2.sort_values("grid_id").reset_index(drop=True))


def test_dbscan_uses_haversine_on_lat_lon_radians():
    """Two points 50 m apart are neighbours; 500 m apart are not (min_samples=2)."""
    df = pd.DataFrame({"latitude": [51.0, 51.00045, 51.0045], "longitude": [-1.0, -1.0, -1.0]})
    labels = fit_dbscan(df, eps_km=0.1, min_samples=2)
    assert labels[0] == labels[1] >= 0 and labels[2] == -1


def test_cluster_stats_reports_largest_share():
    stats = cluster_stats(np.array([0] * 8 + [1] * 2 + [-1] * 5))
    assert stats["clusters"] == 2 and stats["largest_cluster_share"] == 0.8
    assert abs(stats["clustered_fraction"] - 10 / 15) < 1e-12


def test_feature_table_merges_cluster_features_as_static_and_defaults_zero():
    months = pd.date_range("2021-01-01", periods=12, freq="MS")
    rows = [dict(grid_id=g, year_month=m, collision_count=1, total_casualties=1, grid_latitude=51.0, grid_longitude=-1.0)
            for g in ("A", "B") for m in months]
    cf = pd.DataFrame({"grid_id": ["A"], "grid_in_cluster": [1], "cluster_collision_density": [0.7]})
    df = build_feature_table(pd.DataFrame(rows), warmup=0, cluster_features=cf)
    assert (df[df.grid_id == "A"][CLUSTER_FEATURES].nunique() == 1).all()   # constant over time
    assert (df[df.grid_id == "B"][CLUSTER_FEATURES] == 0).all().all()
