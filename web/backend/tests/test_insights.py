"""Tests for the action-list / patterns / regression / tiers / ablations / Folium endpoints.

Like test_api.py these use the real tracked outputs, but they check values against the source files
(never against hard-coded numbers) and cover missing and malformed files: a clean 503, never a traceback.
"""

import ast
import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import data
from app.main import app

client = TestClient(app)
safe = TestClient(app, raise_server_exceptions=False)
APP_DIR = Path(data.__file__).parent


# ---------------------------------------------------------------- action list / cell

def test_action_list_shape_and_labels():
    body = client.get("/api/action-list", params={"n": 5}).json()
    assert len(body["rows"]) == 5
    assert body["status"] == "unvalidated forecast (no outcomes yet)"
    assert body["drivers_note"] == "descriptive, not a model attribution"
    assert body["total"] == body["total_unfiltered"]
    ranks = [r["rank"] for r in body["rows"]]
    assert ranks == sorted(ranks)
    row = body["rows"][0]
    for key in ["grid_id", "risk_tier", "quantile_tier", "probability", "predicted_count_xgboost", "road", "weather",
                "drivers", "summary", "history_12m", "history_months", "shares"]:
        assert key in row
    assert len(row["history_12m"]) == 12 == len(row["history_months"])


def test_action_list_values_come_from_saved_files():
    row = client.get("/api/action-list", params={"n": 1}).json()["rows"][0]
    fc = pd.read_csv(data.FORECAST_PATH).set_index("grid_id").loc[row["grid_id"]]
    reg = pd.read_csv(data.REGRESSION_FORECAST_PATH).set_index("grid_id").loc[row["grid_id"]]
    cells = pd.read_csv(data.FORECAST_CELLS_PATH).set_index("grid_id").loc[row["grid_id"]]
    assert row["probability"] == pytest.approx(fc["probability"])
    assert row["risk_tier"] == fc["risk_tier"]
    assert row["predicted_count_xgboost"] == pytest.approx(reg["predicted_count_xgboost"])
    assert sum(row["history_12m"]) == cells["collisions_last_12_months"]
    drivers = {d["key"]: d for d in row["drivers"]}
    assert drivers["collisions_last_12_months"]["value"] == cells["collisions_last_12_months"]
    assert drivers["historical_avg_collisions"]["percentile"] == pytest.approx(cells["pct_historical_avg"], rel=1e-3)


def test_action_list_percentiles_are_valid_and_never_round_up_to_100():
    rows = client.get("/api/action-list", params={"n": 50}).json()["rows"]
    for r in rows:
        for d in r["drivers"]:
            if d["percentile"] is not None:
                assert 0 <= d["percentile"] < 100
        assert "more than 100%" not in r["summary"]


def test_action_list_filters_are_exact():
    base = client.get("/api/action-list", params={"n": 1}).json()["total"]
    crit = client.get("/api/action-list", params={"n": 500, "tiers": "Critical"}).json()
    assert 0 < crit["total"] < base
    assert {r["risk_tier"] for r in crit["rows"]} == {"Critical"}
    single = client.get("/api/action-list", params={"n": 200, "road": "single"}).json()
    other = client.get("/api/action-list", params={"n": 1, "road": "other"}).json()
    assert single["total"] + other["total"] == base
    assert {r["road"] for r in single["rows"]} == {"single"}
    adverse = client.get("/api/action-list", params={"n": 50, "weather": "adverse"}).json()
    assert {r["weather"] for r in adverse["rows"]} == {"adverse"}
    high = client.get("/api/action-list", params={"n": 50, "quantile_tiers": "HIGH"}).json()
    assert {r["quantile_tier"] for r in high["rows"]} == {"HIGH"}


def test_action_list_share_floor_matches_data():
    first = client.get("/api/action-list", params={"n": 1}).json()
    cut = first["share_percentiles"]["adverse_weather"]["p90"]
    body = client.get("/api/action-list", params={"n": 500, "min_adverse_share": cut}).json()
    assert 0 < body["total"] < first["total_unfiltered"]
    assert all(r["shares"]["adverse_weather"] >= cut for r in body["rows"])


def test_action_list_sort_by_regression_is_descending():
    rows = client.get("/api/action-list", params={"n": 30, "sort": "regression"}).json()["rows"]
    counts = [r["predicted_count_xgboost"] for r in rows]
    assert counts == sorted(counts, reverse=True)


def test_action_list_pagination_and_empty_filter():
    a = client.get("/api/action-list", params={"n": 3}).json()["rows"]
    b = client.get("/api/action-list", params={"n": 3, "offset": 3}).json()["rows"]
    assert {r["grid_id"] for r in a}.isdisjoint({r["grid_id"] for r in b})
    empty = client.get("/api/action-list", params={"tiers": "Nope"}).json()
    assert empty["total"] == 0 and empty["rows"] == []


@pytest.mark.parametrize("params", [{"n": 0}, {"n": 501}, {"offset": -1}, {"road": "bogus"}, {"weather": "bogus"},
                                    {"sort": "bogus"}, {"min_adverse_share": 2}])
def test_action_list_rejects_bad_parameters(params):
    assert client.get("/api/action-list", params=params).status_code == 422


def test_cell_endpoint():
    first = client.get("/api/action-list", params={"n": 1}).json()["rows"][0]
    body = client.get(f"/api/cell/{first['grid_id']}").json()
    assert body["grid_id"] == first["grid_id"] and body["history_12m"] == first["history_12m"]
    assert client.get("/api/cell/does_not_exist").status_code == 404


def test_action_list_still_works_without_regression_forecast(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "REGRESSION_FORECAST_PATH", tmp_path / "absent.csv")
    body = client.get("/api/action-list", params={"n": 2}).json()
    assert body["regression_available"] is False
    assert body["rows"][0]["predicted_count_xgboost"] is None


# ---------------------------------------------------------------- temporal / dbscan

def test_temporal_patterns_consistent_with_file():
    body = client.get("/api/temporal-patterns").json()
    saved = json.loads(data.TEMPORAL_PATH.read_text())
    matrix = body["hour_by_weekday"]
    assert len(matrix) == 7 and all(len(r) == 24 for r in matrix)
    assert sum(map(sum, matrix)) == saved["n_collisions"] == sum(body["by_hour"]) == sum(body["monthly_total"])
    assert body["weekend"]["per_day_ratio_weekend_vs_weekday"] > 0
    assert body["weekday_order"][0] == "Monday"


def test_dbscan_summary_matches_selection_file():
    body = client.get("/api/dbscan/summary").json()
    sel = json.loads(data.DBSCAN_SELECTION_PATH.read_text())
    assert body["stats"] == sel["stats"]
    assert body["chosen"] == {"eps_km": sel["eps_km"], "min_samples": sel["min_samples"]}
    chosen = [r for r in body["sweep"] if r["eps_km"] == sel["eps_km"] and r["min_samples"] == sel["min_samples"]]
    assert len(chosen) == 1 and chosen[0]["clusters"] == sel["stats"]["clusters"]
    assert body["morans_i"]["morans_i"] > body["morans_i"]["expected_under_randomness"]


def test_dbscan_clusters_match_selection_stats():
    body = client.get("/api/dbscan/clusters").json()
    stats = data.load_dbscan_selection()["stats"]
    assert body["total"] == stats["clusters"] == body["returned"]
    sizes = [c["size"] for c in body["clusters"]]
    assert sizes == sorted(sizes, reverse=True) and sizes[0] == stats["largest_cluster_points"]
    assert sum(sizes) == stats["clustered_points"]
    small = client.get("/api/dbscan/clusters", params={"min_size": 100, "limit": 5}).json()
    assert small["returned"] <= 5 and all(c["size"] >= 100 for c in small["clusters"])


# ---------------------------------------------------------------- regression / tiers / ablations

def test_regression_results_labelled_by_split_and_match_file():
    body = client.get("/api/regression/results").json()
    saved = json.loads(data.REGRESSION_METRICS_PATH.read_text())
    assert "validation" in body["validation"]["split_label"] and "test" in body["test"]["split_label"]
    test = {m["model"]: m for m in body["test"]["models"]}
    assert test["xgboost"]["mae"] == saved["test"]["xgboost"]["mae"]
    assert test["historical_average_baseline"]["r2"] == saved["test"]["historical_average_baseline"]["r2"]
    val = {m["model"]: m for m in body["validation"]["models"]}
    assert set(val) == {"xgboost", "random_forest", "historical_average_baseline"}
    gap = body["test"]["mae_gap"]
    assert gap["xgboost_minus_baseline"] == pytest.approx(gap["xgboost_mae"] - gap["baseline_mae"])
    assert gap["relative_difference"] == pytest.approx(gap["xgboost_minus_baseline"] / gap["baseline_mae"])
    assert body["validation"]["mae_gap"]["xgboost_minus_baseline"] < 0 < gap["xgboost_minus_baseline"]


def test_regression_results_without_test_block(monkeypatch, tmp_path):
    saved = json.loads(data.REGRESSION_METRICS_PATH.read_text())
    saved.pop("test")
    path = tmp_path / "m.json"
    path.write_text(json.dumps(saved))
    monkeypatch.setattr(data, "REGRESSION_METRICS_PATH", path)
    body = client.get("/api/regression/results").json()
    assert body["test"] is None and body["validation"]["models"]


def test_regression_forecast_is_labelled_and_verified():
    body = client.get("/api/regression/forecast", params={"n": 10}).json()
    assert body["status"] == "unvalidated forecast (no outcomes yet)"
    assert body["model_hash_verified"] is True
    assert isinstance(body["model"], str) and body["model"] == json.loads(data.REGRESSION_METRICS_PATH.read_text())["selection"]["model_family"]
    counts = [r["predicted_count_xgboost"] for r in body["rows"]]
    assert counts == sorted(counts, reverse=True) and len(counts) == 10
    assert body["total"] == len(pd.read_csv(data.FORECAST_PATH))


def test_quantile_tiers_are_terciles_and_consistent():
    body = client.get("/api/quantile-tiers").json()
    q = pd.read_parquet(data.QUANTILE_TIERS_PATH)
    assert sum(t["cell_months"] for t in body["quantile_tiers"]) == len(q) == sum(t["cell_months"] for t in body["rank_tiers"])
    for t in body["quantile_tiers"]:
        assert t["cell_months"] / len(q) == pytest.approx(1 / 3, abs=0.01)
    rates = [t["hotspot_rate"] for t in body["quantile_tiers"]]
    assert rates == sorted(rates)  # higher tier, higher observed hotspot rate
    assert sum(map(sum, body["crosstab"]["counts"])) == len(q)
    assert sum(t["cells"] for t in body["forecast"]["quantile_tiers"]) == len(pd.read_csv(data.FORECAST_PATH))
    assert body["forecast"]["status"] == "unvalidated forecast (no outcomes yet)"


def test_ablations_report_the_saved_numbers_and_verdicts():
    items = {i["id"]: i for i in client.get("/api/ablations").json()["items"]}
    assert set(items) == {"dbscan_clusters", "neighbor_classifier", "neighbor_regression"}
    for item in items.values():
        assert item["difference"] == pytest.approx(item["with"] - item["without"])
        assert item["helps"] is False
        assert item["decision_rule"]
    reg = json.loads(data.NEIGHBOR_REG_PATH.read_text())
    assert items["neighbor_regression"]["ci95"] == reg["bootstrap_mae_diff_with_minus_without"]["ci95"]
    assert items["neighbor_regression"]["higher_is_better"] is False
    assert items["dbscan_clusters"]["used_in_final_model"] is False


# ---------------------------------------------------------------- Folium map

def test_folium_map_is_served_as_html():
    res = client.get("/api/map/folium")
    assert res.status_code == 200 and res.headers["content-type"].startswith("text/html")
    assert b"leaflet" in res.content.lower()


def test_folium_map_missing_is_503(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "FOLIUM_MAP_PATH", tmp_path / "absent.html")
    res = safe.get("/api/map/folium")
    assert res.status_code == 503 and "hotspot_map.html" in res.json()["detail"]


# ---------------------------------------------------------------- risk map (C3)

def test_risk_map_defaults_to_top_three_tiers_and_reports_totals():
    body = client.get("/api/risk-map").json()
    assert body["tiers"] == ["Critical", "High", "Medium"]
    assert {c["risk_tier"] for c in body["cells"]} <= {"Critical", "High", "Medium"}
    assert body["total_matching"] == body["count"] and body["truncated"] is False
    fc = pd.read_csv(data.FORECAST_PATH)
    assert body["total_matching"] == int(fc["risk_tier"].isin(body["tiers"]).sum())


def test_risk_map_cap_and_truncation_flag():
    body = client.get("/api/risk-map", params={"limit": 100}).json()
    assert body["count"] == 100 and body["truncated"] is True and body["total_matching"] > 100
    scores = [c["score"] for c in body["cells"]]
    assert scores == sorted(scores, reverse=True)
    everything = client.get("/api/risk-map", params={"tiers": "Critical,High,Medium,Low", "limit": 30000}).json()
    # all 30,388 modelled cells exceed the 30,000 hard cap: it must say so rather than silently drop cells
    assert everything["total_matching"] == len(pd.read_csv(data.FORECAST_PATH))
    assert everything["truncated"] is True and everything["count"] == 30000


def test_risk_map_empty_tiers_means_none():
    body = client.get("/api/risk-map", params={"tiers": ""}).json()
    assert body["count"] == 0 and body["tiers"] == []


def test_risk_map_road_weather_filters():
    plain = client.get("/api/risk-map").json()
    single = client.get("/api/risk-map", params={"road": "single"}).json()
    other = client.get("/api/risk-map", params={"road": "other"}).json()
    assert single["road_weather_filters_applied"] is True
    assert single["total_matching"] + other["total_matching"] == plain["total_matching"]
    ids = {c["grid_id"] for c in single["cells"]}
    shares = pd.read_csv(data.FORECAST_CELLS_PATH).set_index("grid_id")
    assert (shares.loc[list(ids), "historical_single_carriageway_share"] > data.SHARE_TAG_THRESHOLD).all()


def test_risk_map_test_source_ignores_road_weather_filters_and_says_so():
    body = client.get("/api/risk-map", params={"source": "test", "road": "single", "limit": 10}).json()
    assert body["road_weather_filters_applied"] is False


# ---------------------------------------------------------------- missing / malformed files -> clean 503

CASES = [
    ("/api/action-list", "FORECAST_CELLS_PATH"),
    ("/api/cell/1059_362", "FORECAST_CELLS_PATH"),
    ("/api/risk-map?road=single", "FORECAST_CELLS_PATH"),
    ("/api/temporal-patterns", "TEMPORAL_PATH"),
    ("/api/dbscan/summary", "DBSCAN_SELECTION_PATH"),
    ("/api/dbscan/summary", "DBSCAN_SWEEP_PATH"),
    ("/api/dbscan/clusters", "DBSCAN_CLUSTERS_PATH"),
    ("/api/regression/results", "REGRESSION_METRICS_PATH"),
    ("/api/regression/forecast", "REGRESSION_FORECAST_PATH"),
    ("/api/quantile-tiers", "QUANTILE_TIERS_PATH"),
    ("/api/ablations", "METRICS_PATH"),
]


def _assert_clean_503(res):
    assert res.status_code == 503, res.text
    assert "detail" in res.json() and "Traceback" not in res.text


@pytest.mark.parametrize("url,attr", CASES)
def test_missing_file_is_clean_503(monkeypatch, tmp_path, url, attr):
    monkeypatch.setattr(data, attr, tmp_path / "absent")
    _assert_clean_503(safe.get(url))


@pytest.mark.parametrize("url,attr", CASES)
@pytest.mark.parametrize("content", [b"{not valid", b"", b"a,b\n1,2\n", b"[]"])
def test_malformed_file_is_clean_503(monkeypatch, tmp_path, url, attr, content):
    bad = tmp_path / "bad"
    bad.write_bytes(content)
    monkeypatch.setattr(data, attr, bad)
    res = safe.get(url)
    # An empty frame / [] can legitimately parse; it must then still not crash.
    assert res.status_code in (200, 503, 404), res.text
    assert "Traceback" not in res.text
    if res.status_code == 503:
        assert "detail" in res.json()


def test_malformed_quantile_parquet_is_clean_503(monkeypatch, tmp_path):
    bad = tmp_path / "q.parquet"
    bad.write_bytes(b"nope")
    monkeypatch.setattr(data, "QUANTILE_TIERS_PATH", bad)
    _assert_clean_503(safe.get("/api/quantile-tiers"))


# ---------------------------------------------------------------- the backend is read-only and model-free

FORBIDDEN_IMPORTS = {"joblib", "pickle", "sklearn", "xgboost", "lightgbm", "torch", "src"}


def test_backend_imports_no_model_or_training_code():
    for py in APP_DIR.glob("*.py"):
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            assert not {n.split(".")[0] for n in names} & FORBIDDEN_IMPORTS, py.name


def test_every_data_path_is_a_tracked_output_not_data_or_models():
    paths = {k: v for k, v in vars(data).items() if k.endswith("_PATH") and isinstance(v, Path)}
    assert len(paths) >= 15  # guard against the scan silently matching nothing
    for name, path in paths.items():
        rel = path.relative_to(data.BASE_DIR).parts
        assert rel[0] in {"results", "outputs"}, f"{name} reads from {rel[0]}/"


def test_backend_serves_only_get_endpoints():
    for route in app.routes:
        methods = getattr(route, "methods", None)
        if methods and route.path.startswith("/api"):
            assert methods <= {"GET", "HEAD"}, route.path
