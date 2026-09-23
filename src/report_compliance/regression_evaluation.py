"""Evaluation metrics and model-selection machinery for the regression track.

All headline metrics (MAE, RMSE, R2) are reported on the RAW COUNT scale.
Models that are fit in log1p space must have their predictions inverted with
expm1() (and clipped at 0, since counts cannot be negative) before being
passed to evaluate_regression().
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def to_raw_scale(log1p_predictions):
    """Invert a log1p-space prediction back to the raw count scale."""
    return np.clip(np.expm1(log1p_predictions), 0, None)


def evaluate_regression(y_true_raw, y_pred_raw):
    """MAE / RMSE / R2 on the raw count scale."""
    y_true_raw = np.asarray(y_true_raw, dtype=float)
    y_pred_raw = np.asarray(y_pred_raw, dtype=float)
    mse = mean_squared_error(y_true_raw, y_pred_raw)
    return {
        "mae": float(mean_absolute_error(y_true_raw, y_pred_raw)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true_raw, y_pred_raw)),
        "n": int(len(y_true_raw)),
        "mean_actual": float(y_true_raw.mean()),
    }


def paired_bootstrap_mae_diff(y_true_raw, pred_a_raw, pred_b_raw, grid_ids, n_boot=300, seed=0):
    """95% CI of MAE(pred_a) - MAE(pred_b), resampling GRIDS with replacement
    (same method as the classifier's paired_bootstrap_ap_diff in
    src/evaluation.py, adapted from average precision to MAE)."""
    y = np.asarray(y_true_raw, dtype=float)
    a = np.asarray(pred_a_raw, dtype=float)
    b = np.asarray(pred_b_raw, dtype=float)
    codes, uniques = pd.factorize(pd.Series(grid_ids))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, len(uniques), len(uniques)), minlength=len(uniques))
        w = counts[codes].astype(float)
        mae_a = np.average(np.abs(y - a), weights=w)
        mae_b = np.average(np.abs(y - b), weights=w)
        diffs.append(mae_a - mae_b)
    diffs = np.array(diffs)
    point_diff = float(mean_absolute_error(y, a) - mean_absolute_error(y, b))
    return {
        "point_diff": point_diff,
        "ci95": [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))],
        "n_bootstrap": n_boot,
    }


def apply_selection_rule(rule, results, val_grid_ids, y_val_raw):
    """Implements a regression analog of the classifier's
    src/train_validate.py::apply_selection_rule: rank candidates by validation
    MAE (lower is better); if the top two are within bootstrap noise, choose
    the simpler by `simplicity_order_simplest_first`; otherwise take the
    lower-MAE candidate.

    `results` : dict name -> {"mae": float, "pred_raw": array, ...}
    """
    ranked = sorted(results, key=lambda f: results[f]["mae"])
    first, second = ranked[0], ranked[1]
    diff = paired_bootstrap_mae_diff(
        y_val_raw, results[first]["pred_raw"], results[second]["pred_raw"], val_grid_ids
    )
    within_noise = diff["ci95"][0] <= 0 <= diff["ci95"][1]
    order = rule["simplicity_order_simplest_first"]
    if within_noise:
        chosen = min((first, second), key=order.index)
        reason = "top two within bootstrap noise; chose the simpler"
    else:
        chosen, reason = first, "top model clearly ahead of the runner-up"
    return chosen, {
        "ranking_by_val_mae": [(f, results[f]["mae"]) for f in ranked],
        "top_two": [first, second],
        "top_two_mae_diff": diff,
        "within_noise": bool(within_noise),
        "chosen": chosen,
        "reason": reason,
    }
