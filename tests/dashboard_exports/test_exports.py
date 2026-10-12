"""Checks on the tracked dashboard exports (outputs/dashboard/*, outputs/regression_forecast_2026-01.*).

They read only tracked files (no data/ needed), so they also pass on a fresh clone. They verify that each
export is internally consistent and agrees with the saved results it was derived from, and that the frozen
artefacts still match the hashes recorded in their own frozen configs.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dashboard_exports._common import DASHBOARD_DIR, OUTPUTS_DIR, PROJECT_ROOT, REGRESSION_FORECAST_META_PATH, REGRESSION_FORECAST_PATH, sha256
from dashboard_exports.forecast_cells import percentile_lower

STATUS = "unvalidated forecast (no outcomes yet)"


def test_percentile_lower_is_strictly_lower_share():
    p = percentile_lower(pd.Series([1, 2, 2, 3]))
    assert p.tolist() == [0.0, 25.0, 25.0, 75.0]
    assert (p < 100).all()


def test_regression_forecast_covers_the_classifier_forecast_cells():
    reg = pd.read_csv(REGRESSION_FORECAST_PATH)
    fc = pd.read_csv(OUTPUTS_DIR / "forecast_2026-01.csv")
    assert set(reg["grid_id"]) == set(fc["grid_id"]) and reg["grid_id"].is_unique
    assert (reg["target_month"] == "2026-01-01").all()
    assert reg[["predicted_count_xgboost", "predicted_count_baseline"]].notna().all().all()
    assert (reg["predicted_count_xgboost"] >= 0).all()
    assert sorted(reg["rank"]) == list(range(1, len(reg) + 1))


def test_regression_forecast_meta_records_unchanged_model_and_unvalidated_status():
    meta = json.loads(REGRESSION_FORECAST_META_PATH.read_text())
    frozen = json.loads((PROJECT_ROOT / "results/regression/regression_frozen_config.json").read_text())
    expected = frozen["hashes_sha256"]["models/regression/final_model.joblib"]
    assert meta["status"] == STATUS
    assert meta["model_sha256_before_load"] == meta["model_sha256_after_load"] == expected
    assert meta["model_hash_matches_frozen_config"] is True
    assert "2025 test rows not read" in meta["inputs"]
    assert meta["regression_config_sha256"] == frozen["config_sha256"]


def test_forecast_cells_history_matches_feature_and_is_descriptive():
    cells = pd.read_csv(DASHBOARD_DIR / "forecast_cells_2026-01.csv")
    meta = json.loads((DASHBOARD_DIR / "forecast_cells_2026-01.meta.json").read_text())
    assert meta["kind"] == "descriptive, not a model attribution"
    assert len(cells) == meta["n_cells"] and cells["grid_id"].is_unique
    hist = cells["history_12m"].str.split(";", expand=True).astype(int)
    assert hist.shape[1] == 12
    assert (hist.sum(axis=1) == cells["collisions_last_12_months"]).all()
    assert meta["history_vs_last12_feature_mismatches"] == 0
    for col in ["pct_historical_avg", "pct_last_12_months"]:
        assert cells[col].between(0, 100, inclusive="left").all()
    share_cols = [c for c in cells.columns if c.endswith("_share")]
    assert len(share_cols) == 7 and cells[share_cols].apply(lambda s: s.between(0, 1)).all().all()
    assert set(cells["quantile_tier"]) == {"LOW", "MEDIUM", "HIGH"}


def test_forecast_cells_quantile_tiers_are_terciles_of_the_classifier_score():
    cells = pd.read_csv(DASHBOARD_DIR / "forecast_cells_2026-01.csv")
    fc = pd.read_csv(OUTPUTS_DIR / "forecast_2026-01.csv").merge(cells[["grid_id", "quantile_tier"]], on="grid_id")
    means = fc.groupby("quantile_tier")["score"].mean()
    assert means["LOW"] < means["MEDIUM"] < means["HIGH"]
    assert (fc["quantile_tier"].value_counts(normalize=True) - 1 / 3).abs().max() < 0.01


def test_dbscan_clusters_match_the_recorded_selection():
    sel = json.loads((PROJECT_ROOT / "results/dbscan_selection.json").read_text())["stats"]
    clusters = pd.read_csv(DASHBOARD_DIR / "dbscan_clusters.csv")
    assert len(clusters) == sel["clusters"]
    assert clusters["size"].max() == sel["largest_cluster_points"]
    assert clusters["size"].sum() == sel["clustered_points"]
    assert clusters["size"].median() == sel["median_cluster_size"]
    assert (clusters["radius_km"] >= 0).all()
    assert clusters["lat"].between(clusters["lat_min"], clusters["lat_max"]).all()


def test_temporal_patterns_internally_consistent():
    t = json.loads((DASHBOARD_DIR / "temporal_patterns.json").read_text())
    m = np.array(t["hour_by_weekday"])
    assert m.shape == (7, 24) and m.sum() == t["n_collisions"]
    assert sum(t["by_hour"]) == sum(t["by_weekday"]) == sum(t["monthly_total"]) == t["n_collisions"]
    assert t["weekend"]["weekend"] + t["weekend"]["weekday"] == t["n_collisions"]
    assert sum(t["time_of_day"]["collisions"]) == t["n_collisions"]
    assert sum(sum(v) for v in t["monthly_by_year"].values()) == t["n_collisions"]
    assert list(t["monthly_by_year"]) == [str(y) for y in t["years"]]


def test_morans_i_file_is_self_consistent():
    m = json.loads((DASHBOARD_DIR / "morans_i.json").read_text())
    assert -1 <= m["morans_i"] <= 1 and 0 < m["p_value"] <= 1
    assert m["n_cells_used"] + m["n_cells_without_modelled_neighbour"] > m["n_cells_used"] > 0
    assert f"{m['morans_i']:.3f}" in m["interpretation"]


@pytest.mark.parametrize("path", sorted(DASHBOARD_DIR.glob("*")) + [REGRESSION_FORECAST_PATH])
def test_each_export_is_small(path):
    assert path.stat().st_size < 5 * 1024 * 1024


def test_frozen_artefacts_match_their_recorded_hashes():
    clf = json.loads((PROJECT_ROOT / "results/frozen_config.json").read_text())["hashes_sha256"]
    for rel in ("models/final_model.joblib", "models/calibrator.joblib"):
        assert sha256(PROJECT_ROOT / rel) == clf[rel], rel
    reg = json.loads((PROJECT_ROOT / "results/regression/regression_frozen_config.json").read_text())["hashes_sha256"]
    assert sha256(PROJECT_ROOT / "models/regression/final_model.joblib") == reg["models/regression/final_model.joblib"]
