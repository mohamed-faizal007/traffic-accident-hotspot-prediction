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
