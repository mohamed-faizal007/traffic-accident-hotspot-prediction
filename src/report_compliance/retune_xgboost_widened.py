"""Edge check: train_validate_regression.py's XGBoost winner (max_depth=4,
learning_rate=0.05, n_estimators=300) sat on the LOWER edge of BOTH tuned
dimensions in the original grid (max_depth in {4,6}, learning_rate in
{0.05,0.1} -- see results/regression/regression_tuning_results.csv). This
re-tunes ONLY xgboost over a widened grid (depth 2-4 below/above the old
boundaries, learning_rate below the old lower boundary), refits the
baseline and the previously-chosen random_forest config once each (not
re-searched) purely to get comparable validation predictions, and re-applies
the SAME pre-registered selection rule (results/regression/regression_frozen_config.json,
still status "registered", unchanged) across all three candidates.

Does not touch the 2025 test set or any frozen classifier file.
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RANDOM_STATE  # noqa: E402
from report_compliance.regression_features import RAW_TARGET_COLUMN, LOG_TARGET_COLUMN, REGRESSION_FEATURE_COLUMNS  # noqa: E402
from report_compliance.regression_baseline import fit_historical_average_baseline, predict_historical_average_baseline  # noqa: E402
from report_compliance.regression_evaluation import to_raw_scale, evaluate_regression, apply_selection_rule  # noqa: E402
from report_compliance.register_regression_selection_rule import REGRESSION_RESULTS_DIR, REGRESSION_FROZEN_CONFIG_PATH, REGRESSION_LOCK_PATH  # noqa: E402
from report_compliance.train_validate_regression import (  # noqa: E402
    load_split_frames, REGRESSION_MODELS_DIR, REGRESSION_METRICS_PATH, REGRESSION_TUNING_CSV,
)

PREVIOUS_RF_BEST_PARAMS = dict(n_estimators=150, max_depth=8, min_samples_leaf=20)

# Original grid: max_depth in (4, 6), learning_rate in (0.05, 0.1); winner
# was max_depth=4 (lower edge), learning_rate=0.05 (lower edge). Widen depth
# 2-3 below the old min (4) and above the old max (6); widen learning_rate
# below its old min (0.05).
XGB_GRID_WIDENED = [
    dict(n_estimators=300, max_depth=d, learning_rate=lr)
    for d in (2, 3, 4, 6, 8) for lr in (0.02, 0.05, 0.1)
]


def tune_xgboost(train, val, features, grid):
    y_val_raw = val[RAW_TARGET_COLUMN].to_numpy()
    rows, best = [], None
    for params in grid:
        model = XGBRegressor(**params, objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(train[features], train[LOG_TARGET_COLUMN])
        pred_raw = to_raw_scale(model.predict(val[features]))
        res = evaluate_regression(y_val_raw, pred_raw)
        rows.append({"family": "xgboost_widened", **params, **res})
        print(f"  xgboost max_depth={params['max_depth']} lr={params['learning_rate']}: val MAE {res['mae']:.4f}")
        if best is None or res["mae"] < best["mae"]:
            best = {"mae": res["mae"], "params": params, "model": model, "pred_raw": pred_raw, "metrics": res}
    return best, rows


def main():
    if REGRESSION_LOCK_PATH.exists():
        sys.exit("The regression track's 2025 test set has already been evaluated.")
    registered = json.loads(REGRESSION_FROZEN_CONFIG_PATH.read_text())
    if registered.get("status") == "frozen":
        sys.exit("regression_frozen_config.json is already frozen.")
    rule = registered["selection_rule"]
    features = REGRESSION_FEATURE_COLUMNS

    frames = load_split_frames()
    train, val = frames["train"], frames["val"]
    y_val_raw = val[RAW_TARGET_COLUMN].to_numpy()

    baseline_fit = fit_historical_average_baseline(train)
    baseline_pred = predict_historical_average_baseline(baseline_fit, val)
    baseline_metrics = evaluate_regression(y_val_raw, baseline_pred)

    print("Refitting random_forest once with its previously-chosen params (not re-searched)...")
    rf_model = RandomForestRegressor(**PREVIOUS_RF_BEST_PARAMS, random_state=RANDOM_STATE, n_jobs=-1)
    rf_model.fit(train[features], train[LOG_TARGET_COLUMN])
    rf_pred = to_raw_scale(rf_model.predict(val[features]))
    rf_metrics = evaluate_regression(y_val_raw, rf_pred)

    print("Tuning xgboost over the widened grid...")
    xgb_best, xgb_rows = tune_xgboost(train, val, features, XGB_GRID_WIDENED)

    old_tuning = pd.read_csv(REGRESSION_TUNING_CSV)
    pd.concat([old_tuning, pd.DataFrame(xgb_rows)], ignore_index=True).to_csv(REGRESSION_TUNING_CSV, index=False)

    depths_tested = sorted({p["max_depth"] for p in XGB_GRID_WIDENED})
    lrs_tested = sorted({p["learning_rate"] for p in XGB_GRID_WIDENED})
    on_edge = (
        xgb_best["params"]["max_depth"] in (depths_tested[0], depths_tested[-1])
        or xgb_best["params"]["learning_rate"] in (lrs_tested[0], lrs_tested[-1])
    )
    print(f"\nWidened-grid xgboost winner: {xgb_best['params']} (val MAE {xgb_best['mae']:.4f})")
    print(f"Still on an edge of the widened grid? {on_edge} (depths tested {depths_tested}, lrs tested {lrs_tested})")

    results = {
        "historical_average_baseline": {"mae": baseline_metrics["mae"], "pred_raw": baseline_pred, "metrics": baseline_metrics},
        "random_forest": {"mae": rf_metrics["mae"], "pred_raw": rf_pred, "metrics": rf_metrics, "params": PREVIOUS_RF_BEST_PARAMS},
        "xgboost": {"mae": xgb_best["mae"], "pred_raw": xgb_best["pred_raw"], "metrics": xgb_best["metrics"], "params": xgb_best["params"]},
    }
    chosen_name, decision = apply_selection_rule(rule, results, val["grid_id"].to_numpy(), y_val_raw)
    print(f"SELECTED: {chosen_name} ({decision['reason']})")

    comparison_table = pd.DataFrame([{"model": name, **r["metrics"]} for name, r in results.items()]).sort_values("mae")
    comparison_table.to_csv(REGRESSION_RESULTS_DIR / "regression_comparison_table.csv", index=False)

    if chosen_name == "random_forest":
        joblib.dump(rf_model, REGRESSION_MODELS_DIR / "final_model.joblib")
    elif chosen_name == "xgboost":
        joblib.dump(xgb_best["model"], REGRESSION_MODELS_DIR / "final_model.joblib")
    else:
        joblib.dump(baseline_fit, REGRESSION_MODELS_DIR / "final_model.joblib")

    metrics = {
        "protocol": "train targets 2021-2023 | validation targets 2024 | test targets 2025 (to be evaluated once, frozen)",
        "selection": {
            "model_family": chosen_name,
            "model_params": results[chosen_name].get("params"),
            "candidate_best_params": {"random_forest": PREVIOUS_RF_BEST_PARAMS, "xgboost": xgb_best["params"]},
            "decision": decision,
            "features": features,
            "target_transform": "log1p(next_month_collisions); metrics reported after expm1() inversion, clipped at 0",
            "xgboost_retuning_note": "original XGB_GRID (depth {4,6} x lr {0.05,0.1}) picked a corner value on both dimensions; "
                                     "widened to depth {2,3,4,6,8} x lr {0.02,0.05,0.1} before freezing",
        },
        "validation": {
            "n_train": int(len(train)),
            "n_val": int(len(val)),
            "all_candidates": {n: r["metrics"] for n, r in results.items()},
            "comparison_table": comparison_table.to_dict(orient="records"),
            "selection_decision": decision,
            "xgboost_edge_check": {
                "original_grid_depths": [4, 6], "original_grid_lrs": [0.05, 0.1],
                "original_winner": {"max_depth": 4, "learning_rate": 0.05},
                "widened_grid_depths": depths_tested, "widened_grid_lrs": lrs_tested,
                "widened_winner": xgb_best["params"],
                "widened_winner_still_on_edge": on_edge,
            },
        },
        "test": None,
    }
    REGRESSION_METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=float))
    print("\n===== VALIDATION (2024 targets) — comparison table (after widened xgboost re-tune) =====")
    print(comparison_table.to_string(index=False))


if __name__ == "__main__":
    main()
