"""Neighbor-cell feature ablation for the regression track -- validation only.

Mirrors src/neighbor_ablation_classifier.py's protocol exactly, but for the
regression track's frozen XGBoost configuration (predicts next-month
collision COUNT, not the binary hotspot label). Refits the same XGBoost
config that is actually frozen
(results/regression/regression_frozen_config.json["model"], read read-only)
with vs without the 3 neighbor-cell features from src/neighbor_features.py,
over 3 seeds (XGBoost has no negative-sampling step the way the classifier
does, so seed-to-seed variance is smaller than the classifier's 5-seed
protocol needs), then a 300-resample grid-bootstrap 95% CI on the
validation MAE difference
(src/report_compliance/regression_evaluation.py::paired_bootstrap_mae_diff).

Reads results/regression/regression_frozen_config.json and the regression
feature table (report_compliance.regression_features, built in memory, never
written to data/processed/regression_features.parquet) read-only. Writes
only to results/regression/neighbor_features_ablation.json (a NEW file --
regression_frozen_config.json, regression_test_evaluated.lock,
regression_metrics.json and models/regression/final_model.joblib are never
touched or reopened for writing). Never loads or scores the 2025 test split.

Pre-registered decision rule (fixed before running, same standard as the
classifier-track ablation): neighbor features are a genuine finding only if
the 95% grid-bootstrap CI of the validation MAE difference
(with_neighbor - without) excludes 0 on the improving (negative) side. See
docs/neighbor_features_plan.md section 4 for the (separate, larger) bar
this project applies before spending the regression track's own one-shot
2025 test evaluation on any such finding.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RESULTS_DIR  # noqa: E402
from neighbor_features import build_neighbor_feature_table, NEIGHBOR_FEATURES  # noqa: E402
from report_compliance.regression_features import (  # noqa: E402
    build_regression_feature_table, REGRESSION_FEATURE_COLUMNS, RAW_TARGET_COLUMN, LOG_TARGET_COLUMN,
)
from report_compliance.regression_evaluation import (  # noqa: E402
    to_raw_scale, evaluate_regression, paired_bootstrap_mae_diff,
)
from report_compliance.register_regression_selection_rule import REGRESSION_RESULTS_DIR  # noqa: E402

REGRESSION_FROZEN_CONFIG_PATH = REGRESSION_RESULTS_DIR / "regression_frozen_config.json"
OUTPUT_PATH = REGRESSION_RESULTS_DIR / "neighbor_features_ablation.json"
REGRESSION_FEATURE_COLUMNS_WITH_NEIGHBOR = REGRESSION_FEATURE_COLUMNS + NEIGHBOR_FEATURES

SEEDS = [42, 0, 1]
N_BOOTSTRAP = 300


def attach_neighbor_features(frame):
    neighbor = build_neighbor_feature_table()
    n_before = len(frame)
    merged = frame.merge(neighbor, on=["grid_id", "year_month"], how="left", validate="one_to_one")
    assert len(merged) == n_before, "neighbor merge changed row count"
    missing = merged[NEIGHBOR_FEATURES].isna().any(axis=1)
    if missing.any():
        raise ValueError(f"{int(missing.sum())} rows had no neighbor-feature match (grid_id/year_month mismatch)")
    return merged


def fit_and_predict(train, val, features, params, seed):
    model = XGBRegressor(**params, objective="reg:squarederror", random_state=seed, n_jobs=-1)
    model.fit(train[features], train[LOG_TARGET_COLUMN])
    return to_raw_scale(model.predict(val[features]))


def neighbor_ablation(train, val, params):
    y_val_raw = val[RAW_TARGET_COLUMN].to_numpy()
    rows, primary = [], {}
    for i, seed in enumerate(SEEDS):
        row = {"seed": seed}
        for name, feats in (("without_neighbor", REGRESSION_FEATURE_COLUMNS),
                             ("with_neighbor", REGRESSION_FEATURE_COLUMNS_WITH_NEIGHBOR)):
            pred = fit_and_predict(train, val, feats, params, seed)
            res = evaluate_regression(y_val_raw, pred)
            row[f"mae_{name}"] = res["mae"]
            row[f"rmse_{name}"] = res["rmse"]
            row[f"r2_{name}"] = res["r2"]
            if i == 0:
                primary[name] = pred
        rows.append(row)
        print(f"seed {seed}: MAE without {row['mae_without_neighbor']:.4f}  with {row['mae_with_neighbor']:.4f}")
    table = pd.DataFrame(rows)
    summary = {}
    for metric in ("mae", "rmse", "r2"):
        diff = table[f"{metric}_with_neighbor"] - table[f"{metric}_without_neighbor"]
        summary[metric] = {
            "without_neighbor_mean": float(table[f"{metric}_without_neighbor"].mean()),
            "with_neighbor_mean": float(table[f"{metric}_with_neighbor"].mean()),
            "diff_mean": float(diff.mean()),
            "diff_sd_over_seeds": float(diff.std(ddof=1)),
            "seeds_where_neighbor_better": int((diff < 0).sum()) if metric in ("mae", "rmse") else int((diff > 0).sum()),
        }
    return table, summary, primary


def main():
    # No lock guard: this script never opens regression_frozen_config.json, regression_metrics.json,
    # regression_test_evaluated.lock, or models/regression/final_model.joblib for writing, at any
    # point. results/regression/regression_frozen_config.json is read-only input below; the only
    # file this script writes is OUTPUT_PATH, a new, additive path.
    frozen = json.loads(REGRESSION_FROZEN_CONFIG_PATH.read_text())  # read-only
    assert frozen["status"] == "frozen"
    assert frozen["model"]["family"] == "xgboost", \
        "frozen regression model is no longer xgboost -- ablation script needs updating"
    params = frozen["model"]["params"]
    print(f"Frozen XGBoost params (read-only, from regression_frozen_config.json): {params}")

    df = build_regression_feature_table()  # in-memory only; never written to regression_features.parquet
    frames = {name: df[df["split"] == name].reset_index(drop=True) for name in ("train", "val")}
    train, val = attach_neighbor_features(frames["train"]), attach_neighbor_features(frames["val"])
    print(f"train rows {len(train):,}; val rows {len(val):,}")

    table, summary, primary = neighbor_ablation(train, val, params)

    y_val_raw = val[RAW_TARGET_COLUMN].to_numpy()
    mae_ci = paired_bootstrap_mae_diff(
        y_val_raw, primary["with_neighbor"], primary["without_neighbor"], val["grid_id"].to_numpy(), n_boot=N_BOOTSTRAP
    )
    neighbor_helps = bool(mae_ci["ci95"][1] < 0)  # MAE: lower is better, so improvement is a NEGATIVE diff

    out = {
        "protocol": "Neighbor-cell feature ablation, regression track. Validation (2024 targets) only; "
                    "2025 test set never touched. Same model family/params as the frozen regression model "
                    "(results/regression/regression_frozen_config.json, read-only); "
                    "no frozen/locked file written to.",
        "params": params,
        "feature_columns_without_neighbor": REGRESSION_FEATURE_COLUMNS,
        "feature_columns_with_neighbor": REGRESSION_FEATURE_COLUMNS_WITH_NEIGHBOR,
        "seeds": table.to_dict(orient="records"),
        "summary": summary,
        "bootstrap_mae_diff_with_minus_without": mae_ci,
        "decision_rule": "neighbor features are a genuine finding only if the 95% grid-bootstrap CI "
                          "of the validation MAE difference (with_neighbor - without) excludes 0 "
                          "on the improving (negative) side",
        "neighbor_features_help": neighbor_helps,
    }
    REGRESSION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(out, indent=2, default=float))

    print("\nNeighbor-feature ablation (mean over seeds):")
    for k, v in summary.items():
        print(f"  {k}: without {v['without_neighbor_mean']:.4f}  with {v['with_neighbor_mean']:.4f}  "
              f"diff {v['diff_mean']:+.4f} (sd {v['diff_sd_over_seeds']:.4f}, better in {v['seeds_where_neighbor_better']}/{len(SEEDS)})")
    print(f"\nBootstrap 95% CI (MAE, with_neighbor - without): point diff {mae_ci['point_diff']:+.4f}  "
          f"CI [{mae_ci['ci95'][0]:+.4f}, {mae_ci['ci95'][1]:+.4f}]")
    print(f"\nneighbor_features_help = {neighbor_helps}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
