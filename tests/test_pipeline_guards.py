"""Grid geometry, risk tiers, model-selection rule and test-set guards."""

import math

import numpy as np
import pandas as pd
import pytest

from evaluation import assign_tiers, paired_bootstrap_ap_diff
from spatial_grid import _TO_WGS, assign_grid, cell_table
from train_validate import apply_selection_rule, load_split_frames


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371008.8 * math.asin(math.sqrt(a))


@pytest.mark.parametrize("lat,lon", [(51.5, -0.12), (53.4, -2.2), (55.9, -3.2)])
def test_grid_cells_are_500m_squares(lat, lon):
    """The old degree grid had cells ~0.29 km wide; the BNG grid must give ~500 m x 500 m."""
    df = assign_grid(pd.DataFrame({"latitude": [lat], "longitude": [lon]}))
    ix, iy = int(df.grid_ix[0]), int(df.grid_iy[0])
    corners = {}
    for name, (cx, cy) in {"sw": (ix, iy), "se": (ix + 1, iy), "nw": (ix, iy + 1)}.items():
        lon_c, lat_c = _TO_WGS.transform(cx * 500.0, cy * 500.0)
        corners[name] = (lat_c, lon_c)
    width = haversine_m(*corners["sw"], *corners["se"])
    height = haversine_m(*corners["sw"], *corners["nw"])
    assert 495 < width < 510 and 495 < height < 510      # ~500 m (BNG scale factor 0.9996 +- convergence)


def test_cell_centre_is_inside_its_cell_and_independent_of_collisions():
    pts = pd.DataFrame({"latitude": [51.5001, 51.5002], "longitude": [-0.1201, -0.1199]})
    grid = assign_grid(pts)
    cells = cell_table(grid)
    assert len(cells) == 1                                # both points share a cell
    again = cell_table(assign_grid(pts.iloc[[0]]))
    assert cells.grid_latitude[0] == again.grid_latitude[0]   # centre does not depend on the points


def test_rank_tiers_per_month():
    n = 1000
    rng = np.random.default_rng(0)
    score = np.concatenate([rng.random(n), rng.random(n)])
    months = np.array(["2025-01"] * n + ["2025-02"] * n)
    tiers = assign_tiers(score, months, {"Critical": 0.01, "High": 0.05, "Medium": 0.10})
    for m in ("2025-01", "2025-02"):
        t = pd.Series(tiers[months == m]).value_counts()
        assert t["Critical"] == 10 and t["High"] == 40 and t["Medium"] == 50 and t["Low"] == 900
    top = np.argsort(-score[:n])[:10]
    assert (tiers[:n][top] == "Critical").all()


def test_training_helpers_never_return_test_rows():
    df = pd.DataFrame({"split": ["train", "val", "test", "unlabeled"], "hotspot": [0, 1, 0, np.nan]})
    assert set(load_split_frames(df)) == {"train", "val"}


def _results(scores_by_family, ap):
    return {f: {"scores": s, "ap": ap[f], "params": {}} for f, s in scores_by_family.items()}


RULE = {"simplicity_order_simplest_first": ["logistic_regression", "hist_gradient_boosting", "random_forest"]}


def test_selection_rule_prefers_simpler_model_within_noise():
    rng = np.random.default_rng(1)
    n = 6000
    y = (rng.random(n) < 0.05).astype(int)
    base = y * 0.5 + rng.random(n)
    val = pd.DataFrame({"grid_id": rng.integers(0, 600, n)})
    res = _results({"random_forest": base, "logistic_regression": base + rng.normal(0, 1e-6, n),
                    "hist_gradient_boosting": rng.random(n)},
                   {"random_forest": 0.30, "logistic_regression": 0.299, "hist_gradient_boosting": 0.05})
    chosen, decision = apply_selection_rule(RULE, res, val, y)
    assert decision["within_noise"] and chosen == "logistic_regression"


def test_selection_rule_takes_clear_winner():
    rng = np.random.default_rng(2)
    n = 6000
    y = (rng.random(n) < 0.05).astype(int)
    good, bad = y * 2.0 + rng.random(n), rng.random(n)
    val = pd.DataFrame({"grid_id": rng.integers(0, 600, n)})
    res = _results({"random_forest": good, "logistic_regression": bad, "hist_gradient_boosting": bad + 1e-3},
                   {"random_forest": 0.9, "logistic_regression": 0.05, "hist_gradient_boosting": 0.05})
    chosen, decision = apply_selection_rule(RULE, res, val, y)
    assert not decision["within_noise"] and chosen == "random_forest"


def test_bootstrap_ci_contains_zero_for_identical_scores():
    rng = np.random.default_rng(3)
    y = (rng.random(3000) < 0.1).astype(int)
    s = rng.random(3000)
    ci = paired_bootstrap_ap_diff(y, s, s, rng.integers(0, 300, 3000), n_boot=50)["ci95"]
    assert ci[0] <= 0 <= ci[1]


def test_test_set_can_only_be_evaluated_once(tmp_path, monkeypatch):
    import evaluate_test
    lock = tmp_path / "test_evaluated.lock"
    lock.write_text("done")
    monkeypatch.setattr(evaluate_test, "LOCK_PATH", lock)
    with pytest.raises(SystemExit):
        evaluate_test.main()


def test_tampered_or_unfrozen_config_is_rejected():
    import evaluate_test
    with pytest.raises(SystemExit):
        evaluate_test.verify_frozen({"status": "selection_rule_only"})
    with pytest.raises(SystemExit):
        evaluate_test.verify_frozen({"status": "frozen", "config_sha256": "x", "a": 1})


def test_frozen_calibrator_loads_and_is_monotone():
    """Regression test: the pickled calibrator refers to __main__.PlattCalibrator."""
    from config import MODELS_DIR
    import evaluate_test
    cal = evaluate_test.load_calibrator(MODELS_DIR / "calibrator.joblib")
    p = cal.predict(np.linspace(0.01, 0.99, 50))
    assert (np.diff(p) >= 0).all() and p.min() >= 0 and p.max() <= 1


@pytest.mark.parametrize("module", ["train_validate", "validation_comparison"])
def test_training_stages_refuse_to_overwrite_after_test_evaluation(module, tmp_path, monkeypatch):
    import importlib
    mod = importlib.import_module(module)
    lock = tmp_path / "lock"
    lock.write_text("done")
    monkeypatch.setattr(mod, "LOCK_PATH", lock)
    with pytest.raises(SystemExit):
        mod.main()
