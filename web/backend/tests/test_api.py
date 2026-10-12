"""Smoke tests for the read-only dashboard API.

Run from web/backend: python -m pytest tests/
These hit the real data files under results/ and outputs/, so they only assert shape/status,
not exact values (the pipeline's own tests in ../../tests/ cover correctness of the numbers).
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_summary():
    res = client.get("/api/summary")
    assert res.status_code == 200
    body = res.json()
    for key in ["model_family", "base_rate", "average_precision", "coverage", "tier_top_fractions"]:
        assert key in body


def test_predictions_default():
    res = client.get("/api/predictions")
    assert res.status_code == 200
    body = res.json()
    assert body["page"] == 1
    assert isinstance(body["rows"], list)


def test_predictions_tier_filter_only_returns_requested_tiers():
    res = client.get("/api/predictions", params={"tiers": "Critical", "page_size": 200})
    assert res.status_code == 200
    rows = res.json()["rows"]
    assert rows, "expected at least one Critical row in the forecast"
    assert all(r["risk_tier"] == "Critical" for r in rows)


def test_predictions_pagination_page_size_respected():
    res = client.get("/api/predictions", params={"page_size": 5})
    assert res.status_code == 200
    assert len(res.json()["rows"]) <= 5


def test_risk_map_forecast_source():
    res = client.get("/api/risk-map", params={"source": "forecast"})
    assert res.status_code == 200
    body = res.json()
    assert body["source"] == "forecast"
    assert body["month"] in body["months"]


def test_risk_map_test_source_has_hotspot_field():
    res = client.get("/api/risk-map", params={"source": "test", "limit": 10})
    assert res.status_code == 200
    body = res.json()
    assert body["count"] <= 10
    if body["cells"]:
        assert "hotspot" in body["cells"][0]


def test_risk_map_invalid_source_rejected():
    res = client.get("/api/risk-map", params={"source": "bogus"})
    assert res.status_code == 422


def test_model_performance():
    res = client.get("/api/model-performance")
    assert res.status_code == 200
    body = res.json()
    assert set(["tn", "fp", "fn", "tp"]).issubset(body["confusion"].keys())
    assert len(body["comparison_rows"]) >= 1


def test_feature_importance():
    res = client.get("/api/feature-importance")
    assert res.status_code == 200
    rows = res.json()["rows"]
    assert rows
    assert set(["feature", "importance", "std"]).issubset(rows[0].keys())


# --- robustness: malformed / missing data files must give a clean 503, never a raw 500 ---

import pytest

from app import data

safe_client = TestClient(app, raise_server_exceptions=False)


def test_predictions_grid_id_is_a_literal_not_a_regex():
    for needle in ["(", "[", "*", "\\"]:
        res = client.get("/api/predictions", params={"grid_id": needle})
        assert res.status_code == 200, needle
        assert res.json()["total"] == 0


def test_predictions_unknown_tier_and_far_page_are_empty_not_errors():
    assert client.get("/api/predictions", params={"tiers": "Bogus"}).json()["total"] == 0
    assert client.get("/api/predictions", params={"page": 999999}).json()["rows"] == []


@pytest.mark.parametrize("content", ["{bad json", "", "{}", "[]", "null"])
def test_malformed_metrics_returns_503_with_message(monkeypatch, tmp_path, content):
    bad = tmp_path / "metrics.json"
    bad.write_text(content)
    monkeypatch.setattr(data, "METRICS_PATH", bad)
    for url in ["/api/summary", "/api/model-performance"]:
        res = safe_client.get(url)
        assert res.status_code == 503, (url, content, res.text)
        assert "detail" in res.json()
        assert "Traceback" not in res.text


def test_missing_metrics_returns_503(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "METRICS_PATH", tmp_path / "absent.json")
    res = safe_client.get("/api/summary")
    assert res.status_code == 503
    assert "metrics.json" in res.json()["detail"]


def test_corrupt_forecast_csv_returns_503(monkeypatch, tmp_path):
    bad = tmp_path / "forecast.csv"
    bad.write_text("a,b\n1,2\n")  # parses, but lacks the expected columns
    monkeypatch.setattr(data, "FORECAST_PATH", bad)
    for url in ["/api/predictions", "/api/risk-map?source=forecast"]:
        res = safe_client.get(url)
        assert res.status_code == 503, url
        assert "Traceback" not in res.text


def test_unparseable_parquet_returns_503(monkeypatch, tmp_path):
    bad = tmp_path / "test_predictions.parquet"
    bad.write_bytes(b"not a parquet file")
    monkeypatch.setattr(data, "TEST_PRED_PATH", bad)
    res = safe_client.get("/api/risk-map", params={"source": "test"})
    assert res.status_code == 503
    assert "PARQUET" in res.json()["detail"]


def test_corrupt_feature_importance_returns_503(monkeypatch, tmp_path):
    bad = tmp_path / "fi.csv"
    bad.write_bytes(b"\x00\x01\x02")
    monkeypatch.setattr(data, "IMPORTANCE_PATH", bad)
    assert safe_client.get("/api/feature-importance").status_code in (503,)
