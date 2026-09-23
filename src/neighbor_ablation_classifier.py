"""Neighbor-cell feature ablation for the FROZEN classifier -- validation only.

Refits the same HistGradientBoostingClassifier configuration that is actually
frozen (results/metrics.json["selection"]["model_family"] ==
"hist_gradient_boosting", params read read-only from that file), with vs
without the 3 neighbor-cell features from src/neighbor_features.py, over 5
seeds (same protocol as src/validation_comparison.py::cluster_ablation for
the DBSCAN cluster features), then a 300-resample grid-bootstrap 95% CI on
the validation AP difference (src/validation_comparison.py::bootstrap).

Reads results/frozen_config.json, results/metrics.json, and
data/processed/ml_features.parquet read-only. Writes only to
results/neighbor_features_ablation.json (a NEW file -- metrics.json itself,
frozen_config.json, test_evaluated.lock, final_model.joblib and
calibrator.joblib are never touched or reopened for writing). Never loads
or scores the 2025 test split.

Pre-registered decision rule (fixed before running): neighbor features are a
genuine finding only if the 95% grid-bootstrap CI of the validation AP
difference (with_neighbor - without) excludes 0 on the positive side. See
docs/neighbor_features_plan.md section 4 for the (separate, larger) bar
this project applies before spending the one-shot 2025 test evaluation on
any such finding.
"""

import json

import numpy as np
import pandas as pd

from config import METRICS_PATH, RESULTS_DIR, RANDOM_STATE
from evaluation import evaluate_scores
from features import FEATURE_COLUMNS
from neighbor_features import build_neighbor_feature_table, NEIGHBOR_FEATURES
from train_validate import fit_hgb, load_split_frames, sample_training, score
from validation_comparison import bootstrap, N_SEEDS

OUTPUT_PATH = RESULTS_DIR / "neighbor_features_ablation.json"
FEATURE_COLUMNS_WITH_NEIGHBOR = FEATURE_COLUMNS + NEIGHBOR_FEATURES


def attach_neighbor_features(frame):
    neighbor = build_neighbor_feature_table()
    n_before = len(frame)
    merged = frame.merge(neighbor, on=["grid_id", "year_month"], how="left", validate="one_to_one")
    assert len(merged) == n_before, "neighbor merge changed row count"
    missing = merged[NEIGHBOR_FEATURES].isna().any(axis=1)
    if missing.any():
        raise ValueError(f"{int(missing.sum())} rows had no neighbor-feature match (grid_id/year_month mismatch)")
    return merged


def neighbor_ablation(train, val, params):
    y = val["hotspot"].to_numpy().astype(int)
    months = val["target_month"].to_numpy()
    rows, primary = [], {}
    for i, seed in enumerate([RANDOM_STATE] + list(range(N_SEEDS - 1))):
        train_s = sample_training(train, seed=seed)
        row = {"seed": seed}
        for name, feats in (("without_neighbor", FEATURE_COLUMNS), ("with_neighbor", FEATURE_COLUMNS_WITH_NEIGHBOR)):
            model = fit_hgb(train_s, feats, params, seed=seed)
            s = score(model, val, feats)
            res = evaluate_scores(y, s, months)
            row[f"ap_{name}"] = res["average_precision"]
            row[f"top1_{name}"] = res["at_top_k_per_month"]["top_1pct"]["precision"]
            row[f"top5_{name}"] = res["at_top_k_per_month"]["top_5pct"]["precision"]
            if i == 0:
                primary[name] = s
        rows.append(row)
        print(f"seed {seed}: AP without {row['ap_without_neighbor']:.4f}  with {row['ap_with_neighbor']:.4f}")
    table = pd.DataFrame(rows)
    summary = {}
    for metric in ("ap", "top1", "top5"):
        diff = table[f"{metric}_with_neighbor"] - table[f"{metric}_without_neighbor"]
        summary[metric] = {
            "without_neighbor_mean": float(table[f"{metric}_without_neighbor"].mean()),
            "with_neighbor_mean": float(table[f"{metric}_with_neighbor"].mean()),
            "diff_mean": float(diff.mean()),
            "diff_sd_over_seeds": float(diff.std(ddof=1)),
            "seeds_where_neighbor_better": int((diff > 0).sum()),
        }
    return table, summary, primary


def main():
    # No LOCK_PATH guard here (unlike train_validate.py/validation_comparison.py): this script
    # never opens frozen_config.json, metrics.json, test_evaluated.lock, or either model artifact
    # for writing, at any point, regardless of whether the 2025 test set has been evaluated
    # (results/metrics.json is read-only input below; the only file this script writes is
    # OUTPUT_PATH, a new, additive path). The lock guard in those other scripts exists to stop
    # a re-run from silently overwriting frozen results -- there is nothing here for it to protect.
    metrics = json.loads(METRICS_PATH.read_text())  # read-only: never written back to
    assert metrics["selection"]["model_family"] == "hist_gradient_boosting", \
        "frozen classifier is no longer hist_gradient_boosting -- ablation script needs updating"
    params = metrics["selection"]["model_params"]
    print(f"Frozen HGB params (read-only, from results/metrics.json): {params}")

    frames = load_split_frames()
    train, val = attach_neighbor_features(frames["train"]), attach_neighbor_features(frames["val"])
    print(f"train rows {len(train):,} (pos {int(train.hotspot.sum()):,}); val rows {len(val):,} (pos {int(val.hotspot.sum()):,})")

    table, summary, primary = neighbor_ablation(train, val, params)

    stored = pd.read_parquet(RESULTS_DIR / "val_scores.parquet")  # read-only
    assert (stored["grid_id"].to_numpy() == val["grid_id"].to_numpy()).all(), \
        "val row order differs from val_scores.parquet -- neighbor merge must have reordered rows"
    scores = {
        "hist_gradient_boosting": stored["hist_gradient_boosting"].to_numpy(),
        "hist_gradient_boosting_with_neighbor": primary["with_neighbor"],
    }
    boot = bootstrap(val, scores, comparisons=[("hist_gradient_boosting_with_neighbor", "hist_gradient_boosting")])

    ap_ci = boot["comparisons"]["hist_gradient_boosting_with_neighbor - hist_gradient_boosting"]["ap"]["ci95"]
    neighbor_helps = bool(ap_ci[0] > 0)

    out = {
        "protocol": "Neighbor-cell feature ablation, classifier track. Validation (2024 targets) only; "
                    "2025 test set never touched. Same model family/params as the frozen classifier "
                    "(results/metrics.json, read-only); no frozen/locked file written to.",
        "params": params,
        "feature_columns_without_neighbor": FEATURE_COLUMNS,
        "feature_columns_with_neighbor": FEATURE_COLUMNS_WITH_NEIGHBOR,
        "seeds": table.to_dict(orient="records"),
        "summary": summary,
        "bootstrap": boot,
        "decision_rule": "neighbor features are a genuine finding only if the 95% grid-bootstrap CI "
                          "of the validation AP difference (with_neighbor - without) excludes 0 on the positive side",
        "neighbor_features_help": neighbor_helps,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(out, indent=2, default=float))

    print("\nNeighbor-feature ablation (mean over seeds):")
    for k, v in summary.items():
        print(f"  {k}: without {v['without_neighbor_mean']:.4f}  with {v['with_neighbor_mean']:.4f}  "
              f"diff {v['diff_mean']:+.4f} (sd {v['diff_sd_over_seeds']:.4f}, better in {v['seeds_where_neighbor_better']}/{N_SEEDS})")
    print("\nBootstrap 95% CI (AP, with_neighbor - without):")
    print(f"  point diff {boot['comparisons']['hist_gradient_boosting_with_neighbor - hist_gradient_boosting']['ap']['point_diff']:+.4f}  "
          f"CI [{ap_ci[0]:+.4f}, {ap_ci[1]:+.4f}]")
    print(f"\nneighbor_features_help = {neighbor_helps}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
