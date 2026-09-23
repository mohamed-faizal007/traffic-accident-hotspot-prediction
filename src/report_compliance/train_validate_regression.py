"""Train on 2021-2023 targets; tune and select on 2024 targets, for the
regression track (predicts next-month collision COUNT, not the binary
hotspot label). The 2025 test set is NEVER touched here -- see
evaluate_regression_test.py, to be run once, only after this validation
step is reviewed and approved.

Mirrors src/train_validate.py's discipline (pre-registered selection rule,
train/val split by target month, tuning on validation only) but is an
entirely separate, additive script: it does not import from, call, or
write to anything in src/train_validate.py, src/freeze_config.py,
src/evaluate_test.py, models/final_model.joblib, models/calibrator.joblib,
results/frozen_config.json, results/metrics.json, or results/test_evaluated.lock.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RANDOM_STATE, RESULTS_DIR, MODELS_DIR  # noqa: E402
from report_compliance.regression_features import (  # noqa: E402
    build_regression_feature_table, REGRESSION_FEATURE_COLUMNS, RAW_TARGET_COLUMN, LOG_TARGET_COLUMN,
)
from report_compliance.regression_baseline import (  # noqa: E402
    fit_historical_average_baseline, predict_historical_average_baseline,
)
from report_compliance.regression_evaluation import (  # noqa: E402
    to_raw_scale, evaluate_regression, apply_selection_rule,
)
from report_compliance.register_regression_selection_rule import (  # noqa: E402
    REGRESSION_RESULTS_DIR, REGRESSION_FROZEN_CONFIG_PATH, REGRESSION_LOCK_PATH,
)

REGRESSION_MODELS_DIR = MODELS_DIR / "regression"
REGRESSION_METRICS_PATH = REGRESSION_RESULTS_DIR / "regression_metrics.json"
REGRESSION_TUNING_CSV = REGRESSION_RESULTS_DIR / "regression_tuning_results.csv"

RF_GRID = [
    dict(n_estimators=150, max_depth=d, min_samples_leaf=leaf)
    for d in (8, 12, 16) for leaf in (5, 20)
]
XGB_GRID = [
    dict(n_estimators=300, max_depth=d, learning_rate=lr)
    for d in (4, 6) for lr in (0.05, 0.1)
]


def load_split_frames():
    df = build_regression_feature_table()
    return {name: df[df["split"] == name].reset_index(drop=True) for name in ("train", "val", "test")}


def tune_random_forest(train, val, features, grid):
    y_val_raw = val[RAW_TARGET_COLUMN].to_numpy()
    rows, best = [], None
    for params in grid:
        model = RandomForestRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(train[features], train[LOG_TARGET_COLUMN])
        pred_raw = to_raw_scale(model.predict(val[features]))
        res = evaluate_regression(y_val_raw, pred_raw)
        rows.append({"family": "random_forest", **params, **res})
        if best is None or res["mae"] < best["mae"]:
            best = {"mae": res["mae"], "params": params, "model": model, "pred_raw": pred_raw, "metrics": res}
    print(f"random_forest: best val MAE {best['mae']:.4f} with {best['params']}")
    return best, rows


def tune_xgboost(train, val, features, grid):
    y_val_raw = val[RAW_TARGET_COLUMN].to_numpy()
    rows, best = [], None
    for params in grid:
        model = XGBRegressor(**params, objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(train[features], train[LOG_TARGET_COLUMN])
        pred_raw = to_raw_scale(model.predict(val[features]))
        res = evaluate_regression(y_val_raw, pred_raw)
        rows.append({"family": "xgboost", **params, **res})
        if best is None or res["mae"] < best["mae"]:
            best = {"mae": res["mae"], "params": params, "model": model, "pred_raw": pred_raw, "metrics": res}
    print(f"xgboost: best val MAE {best['mae']:.4f} with {best['params']}")
    return best, rows


def main():
    if REGRESSION_LOCK_PATH.exists():
        sys.exit("The regression track's 2025 test set has been evaluated; results and models are frozen. Not overwriting.")
    if not REGRESSION_FROZEN_CONFIG_PATH.exists():
        sys.exit("Selection rule not registered; run register_regression_selection_rule.py first.")

    registered = json.loads(REGRESSION_FROZEN_CONFIG_PATH.read_text())
    if registered.get("status") == "frozen":
        sys.exit("regression_frozen_config.json is already frozen; re-running tuning would invalidate it.")
    rule = registered["selection_rule"]
    features = REGRESSION_FEATURE_COLUMNS

    REGRESSION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REGRESSION_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    frames = load_split_frames()
    train, val = frames["train"], frames["val"]
    y_val_raw = val[RAW_TARGET_COLUMN].to_numpy()
    print(f"train rows {len(train):,}; val rows {len(val):,}")

    baseline_fit = fit_historical_average_baseline(train)
    baseline_pred = predict_historical_average_baseline(baseline_fit, val)
    baseline_metrics = evaluate_regression(y_val_raw, baseline_pred)
    print(f"historical_average_baseline: val MAE {baseline_metrics['mae']:.4f}")

    tuning_rows = []
    rf_best, rf_rows = tune_random_forest(train, val, features, RF_GRID)
    xgb_best, xgb_rows = tune_xgboost(train, val, features, XGB_GRID)
    tuning_rows += rf_rows + xgb_rows
    pd.DataFrame(tuning_rows).to_csv(REGRESSION_TUNING_CSV, index=False)

    results = {
        "historical_average_baseline": {"mae": baseline_metrics["mae"], "pred_raw": baseline_pred, "metrics": baseline_metrics},
        "random_forest": {"mae": rf_best["mae"], "pred_raw": rf_best["pred_raw"], "metrics": rf_best["metrics"], "params": rf_best["params"]},
        "xgboost": {"mae": xgb_best["mae"], "pred_raw": xgb_best["pred_raw"], "metrics": xgb_best["metrics"], "params": xgb_best["params"]},
    }
    chosen_name, decision = apply_selection_rule(rule, results, val["grid_id"].to_numpy(), y_val_raw)
    print(f"SELECTED: {chosen_name} ({decision['reason']})")

    comparison_table = pd.DataFrame([
        {"model": name, **r["metrics"]} for name, r in results.items()
    ]).sort_values("mae")
    comparison_table.to_csv(REGRESSION_RESULTS_DIR / "regression_comparison_table.csv", index=False)

    if chosen_name == "random_forest":
        joblib.dump(rf_best["model"], REGRESSION_MODELS_DIR / "final_model.joblib")
    elif chosen_name == "xgboost":
        joblib.dump(xgb_best["model"], REGRESSION_MODELS_DIR / "final_model.joblib")
    else:
        joblib.dump(baseline_fit, REGRESSION_MODELS_DIR / "final_model.joblib")  # dict lookup, not an sklearn estimator

    validation = {
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "all_candidates": {n: r["metrics"] for n, r in results.items()},
        "comparison_table": comparison_table.to_dict(orient="records"),
        "selection_decision": decision,
    }
    selection = {
        "model_family": chosen_name,
        "model_params": results[chosen_name].get("params"),
        "candidate_best_params": {
            "random_forest": rf_best["params"],
            "xgboost": xgb_best["params"],
        },
        "decision": decision,
        "features": features,
        "target_transform": "log1p(next_month_collisions); metrics reported after expm1() inversion, clipped at 0",
    }
    metrics = {
        "protocol": "train targets 2021-2023 | validation targets 2024 | test targets 2025 (to be evaluated once, frozen)",
        "selection": selection,
        "validation": validation,
        "test": None,
    }
    REGRESSION_METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=float))
    print("\n===== VALIDATION (2024 targets) — comparison table =====")
    print(comparison_table.to_string(index=False))


if __name__ == "__main__":
    main()
