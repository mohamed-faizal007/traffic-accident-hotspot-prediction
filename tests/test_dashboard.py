"""Run every dashboard page headlessly and check displayed numbers against results/metrics.json."""

import json
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
METRICS = json.loads((ROOT / "results" / "metrics.json").read_text())
PAGES = ["Home", "Hotspot Predictions", "Risk Map", "Model Performance", "Feature Importance"]

needs_outputs = pytest.mark.skipif(METRICS.get("test") is None, reason="2025 outputs not generated")


def run_page(page, **radio):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    at.sidebar.radio[0].set_value(page).run()
    for label, value in radio.items():
        next(r for r in at.radio if r.label == label).set_value(value).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


@needs_outputs
@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(page):
    run_page(page)


@needs_outputs
def test_home_metrics_match_metrics_json():
    at = run_page("Home")
    t = METRICS["test"]["score_raw"]
    shown = {m.label: m.value for m in at.metric}
    assert shown["Base rate (hotspot cell-months)"] == f"{100 * t['base_rate']:.2f}%"
    assert shown["Average precision"] == f"{t['average_precision']:.3f}"
    assert shown["Precision in top 1% each month"] == f"{100 * t['at_top_k_per_month']['top_1pct']['precision']:.1f}%"
    tb = METRICS["test"]["baselines"]["trailing_12_month_count"]["at_top_k_per_month"]["top_1pct"]["precision"]
    assert shown["Same, trailing-12-month baseline"] == f"{100 * tb:.1f}%"
    cov = METRICS["test"]["coverage"]["coverage"]
    assert any(f"{100 * cov:.1f}%" in w.value for w in at.warning)


@needs_outputs
def test_model_performance_table_matches_metrics_json():
    at = run_page("Model Performance")
    t = METRICS["test"]
    table = at.dataframe[0].value
    frozen_row = table.iloc[0]
    assert abs(frozen_row["Avg precision"] - t["score_raw"]["average_precision"]) < 1e-9
    assert abs(frozen_row["Precision @ top 1%"] - t["score_raw"]["at_top_k_per_month"]["top_1pct"]["precision"]) < 1e-9
    names = list(table["Method"])
    assert any("persistence" in n for n in names) and any("trailing" in n for n in names)
    base = t["baselines"]["trailing_12_month_count"]
    assert abs(table.iloc[2]["Avg precision"] - base["average_precision"]) < 1e-9
    text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    conf = t["score_raw"]["confusion"]
    assert f"Precision {conf['precision']:.3f}, recall {conf['recall']:.3f}, F1 {conf['f1']:.3f}" in text


@needs_outputs
def test_map_and_forecast_pages_use_saved_outputs_and_both_map_sources():
    run_page("Hotspot Predictions")
    run_page("Risk Map")
    run_page("Risk Map", **{"Data": "2025 retrospective (test predictions)"})


def test_no_hardcoded_metrics_in_app_source():
    import re
    src = (ROOT / "app.py").read_text()
    numbers = re.findall(r"(?<![\w.])0\.\d{3,}", src)
    assert not numbers, f"hard-coded metric-like numbers in app.py: {numbers}"
