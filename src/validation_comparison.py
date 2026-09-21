"""Validation-only comparisons (2024 targets). Never touches 2025.

1. Cluster-feature ablation: same RF hyper-parameters (chosen WITHOUT cluster features),
   with vs without the DBSCAN cluster features, over several seeds.
2. Grid-level bootstrap CIs for the differences between RF, logistic regression and the
   trailing-12-month baseline (and RF+cluster vs RF).

Decision rule fixed in advance: the cluster features would be kept in the final feature set
only if the 95% bootstrap CI for the validation AP difference (RF+cluster - RF) excludes 0
on the positive side. Run AFTER train_validate.py (needs results/val_scores.parquet).
"""

import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from config import METRICS_PATH, RESULTS_DIR, RANDOM_STATE
from evaluation import evaluate_scores, top_k_mask
from freeze_config import LOCK_PATH
from features import FEATURE_COLUMNS, FEATURE_COLUMNS_WITH_CLUSTER
from train_validate import fit_rf, load_split_frames, sample_training, score

N_SEEDS = 5
N_BOOTSTRAP = 300
TOP_FRACTIONS = {"top_1pct": 0.01, "top_5pct": 0.05}


def cluster_ablation(train, val, params):
    y = val["hotspot"].to_numpy().astype(int)
    months = val["target_month"].to_numpy()
    rows, primary = [], {}
    for i, seed in enumerate([RANDOM_STATE] + list(range(N_SEEDS - 1))):
        train_s = sample_training(train, seed=seed)
        row = {"seed": seed}
        for name, feats in (("without_cluster", FEATURE_COLUMNS), ("with_cluster", FEATURE_COLUMNS_WITH_CLUSTER)):
            s = score(fit_rf(train_s, feats, params, seed=seed), val, feats)
            res = evaluate_scores(y, s, months)
            row[f"ap_{name}"] = res["average_precision"]
            row[f"top1_{name}"] = res["at_top_k_per_month"]["top_1pct"]["precision"]
            row[f"top5_{name}"] = res["at_top_k_per_month"]["top_5pct"]["precision"]
            if i == 0:
                primary[name] = s
        rows.append(row)
        print(f"seed {seed}: AP without {row['ap_without_cluster']:.4f}  with {row['ap_with_cluster']:.4f}")
    table = pd.DataFrame(rows)
    summary = {}
    for metric in ("ap", "top1", "top5"):
        diff = table[f"{metric}_with_cluster"] - table[f"{metric}_without_cluster"]
        summary[metric] = {
            "without_cluster_mean": float(table[f"{metric}_without_cluster"].mean()),
            "with_cluster_mean": float(table[f"{metric}_with_cluster"].mean()),
            "diff_mean": float(diff.mean()),
            "diff_sd_over_seeds": float(diff.std(ddof=1)),
            "seeds_where_cluster_better": int((diff > 0).sum()),
        }
    return table, summary, primary


def bootstrap(val, scores, n_boot=N_BOOTSTRAP, seed=0, comparisons=None):
    """Resample GRIDS with replacement (keeps a grid's 12 months together)."""
    y = val["hotspot"].to_numpy().astype(int)
    months = val["target_month"].to_numpy()
    grid_codes, uniques = pd.factorize(val["grid_id"])
    n_grids = len(uniques)
    masks = {name: {k: top_k_mask(s, months, f) for k, f in TOP_FRACTIONS.items()} for name, s in scores.items()}
    rng = np.random.default_rng(seed)

    def stats(w):
        out = {}
        for name, s in scores.items():
            out[(name, "ap")] = average_precision_score(y, s, sample_weight=w)
            for k, m in masks[name].items():
                out[(name, k)] = (w[m] * y[m]).sum() / w[m].sum()
        return out

    point = stats(np.ones(len(y)))
    reps = []
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, n_grids, n_grids), minlength=n_grids)
        reps.append(stats(counts[grid_codes].astype(float)))
    reps = pd.DataFrame(reps)
    reps.columns = [f"{a}|{b}" for a, b in reps.columns]

    comparisons = comparisons or [("random_forest", "logistic_regression"), ("hist_gradient_boosting", "logistic_regression"),
                   ("random_forest", "hist_gradient_boosting"),
                   ("random_forest", "trailing_12_month"), ("hist_gradient_boosting", "trailing_12_month"),
                   ("logistic_regression", "trailing_12_month"), ("random_forest_with_cluster", "random_forest")]
    out = {"n_bootstrap": n_boot, "unit": "grid (resampled with replacement)", "comparisons": {}}
    for a, b in comparisons:
        entry = {}
        for metric in ("ap", "top_1pct", "top_5pct"):
            d = reps[f"{a}|{metric}"] - reps[f"{b}|{metric}"]
            entry[metric] = {
                "point_diff": float(point[(a, metric)] - point[(b, metric)]),
                "ci95": [float(d.quantile(0.025)), float(d.quantile(0.975))],
                "share_of_resamples_positive": float((d > 0).mean()),
            }
        out["comparisons"][f"{a} - {b}"] = entry
    out["point_estimates"] = {f"{n}|{m}": float(point[(n, m)]) for (n, m) in point}
    return out


def main():
    if LOCK_PATH.exists():
        sys.exit("The 2025 test set has been evaluated; results and models are frozen. Not overwriting.")
    metrics = json.loads(METRICS_PATH.read_text())
    params = metrics["selection"]["candidate_best_params"]["random_forest"]
    frames = load_split_frames()
    train, val = frames["train"], frames["val"]
    print(f"RF params (chosen without cluster features): {params}")

    table, summary, primary = cluster_ablation(train, val, params)
    stored = pd.read_parquet(RESULTS_DIR / "val_scores.parquet")
    assert (stored["grid_id"].to_numpy() == val["grid_id"].to_numpy()).all()
    scores = {
        "random_forest": stored["random_forest"].to_numpy(),
        "random_forest_with_cluster": primary["with_cluster"],
        "hist_gradient_boosting": stored["hist_gradient_boosting"].to_numpy(),
        "logistic_regression": stored["logistic_regression"].to_numpy(),
        "trailing_12_month": val["base_trailing_12"].to_numpy(),
    }
    boot = bootstrap(val, scores)

    ap_ci = boot["comparisons"]["random_forest_with_cluster - random_forest"]["ap"]["ci95"]
    metrics["validation"]["cluster_ablation"] = {
        "params": params, "seeds": table.to_dict(orient="records"), "summary": summary,
        "decision_rule": "keep cluster features only if the 95% grid-bootstrap CI of the val AP difference excludes 0 on the positive side",
        "cluster_features_help": bool(ap_ci[0] > 0),
        "used_in_final_model": False,
    }
    metrics["validation"]["bootstrap"] = boot
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=float))

    print("\nCluster ablation (mean over seeds):")
    for k, v in summary.items():
        print(f"  {k}: without {v['without_cluster_mean']:.4f}  with {v['with_cluster_mean']:.4f}  diff {v['diff_mean']:+.4f} (sd {v['diff_sd_over_seeds']:.4f}, better in {v['seeds_where_cluster_better']}/{N_SEEDS})")
    print("\nBootstrap 95% CIs:")
    for name, entry in boot["comparisons"].items():
        for metric, e in entry.items():
            print(f"  {name:55s} {metric:9s} {e['point_diff']:+.4f}  [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}]")


if __name__ == "__main__":
    main()
