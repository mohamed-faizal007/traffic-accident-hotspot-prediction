"""Evaluation metrics for rare-event hotspot ranking.

ROC-AUC is reported for reference only; average precision (PR-AUC), lift over the
base rate, and per-month precision@k / recall@k are the headline metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)

TOP_FRACTIONS = (0.01, 0.05, 0.10)


def _order(score, rng):
    """Descending order with random tie-breaking (many grids share identical scores)."""
    return np.lexsort((rng.random(len(score)), -np.asarray(score, dtype=float)))


def top_k_mask(score, months, frac, seed=0):
    """Boolean mask of the top `frac` grids within each month (random tie-breaking)."""
    score = np.asarray(score, dtype=float)
    months = np.asarray(months)
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(score), dtype=bool)
    for m in np.unique(months):
        idx = np.where(months == m)[0]
        k = max(1, int(np.ceil(frac * len(idx))))
        mask[idx[_order(score[idx], rng)[:k]]] = True
    return mask


def precision_recall_at_k(y, score, months, fractions=TOP_FRACTIONS, seed=0):
    """Rank grids within each month, take the top fraction, pool the counts."""
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    months = np.asarray(months)
    rng = np.random.default_rng(seed)
    base = y.mean()
    out = {}
    for frac in fractions:
        tp = picked = 0
        for m in np.unique(months):
            idx = np.where(months == m)[0]
            k = max(1, int(np.ceil(frac * len(idx))))
            top = idx[_order(score[idx], rng)[:k]]
            tp += int(y[top].sum())
            picked += k
        precision = tp / picked
        out[f"top_{int(frac * 100)}pct"] = {
            "precision": precision,
            "recall": tp / max(int(y.sum()), 1),
            "lift": precision / base if base > 0 else float("nan"),
            "grids_flagged_per_month": picked / len(np.unique(months)),
        }
    return out


def per_month_ap(y, score, months):
    y, score, months = map(np.asarray, (y, score, months))
    rows = {}
    for m in np.unique(months):
        mask = months == m
        label = str(pd.Timestamp(m).strftime("%Y-%m"))
        rows[label] = {
            "n": int(mask.sum()),
            "positives": int(y[mask].sum()),
            "average_precision": float(average_precision_score(y[mask], score[mask])),
        }
    return rows


def confusion_at(y, score, threshold):
    pred = (np.asarray(score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": precision, "recall": recall, "f1": f1,
        "flagged_fraction": float(pred.mean()),
    }


def evaluate_scores(y, score, months, probability=False, threshold=None):
    """Full metric bundle for one scoring method. Brier only if `probability`."""
    y = np.asarray(y).astype(int)
    base = float(y.mean())
    ap = float(average_precision_score(y, score))
    result = {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "base_rate": base,
        "average_precision": ap,
        "ap_lift_over_base_rate": ap / base,
        "roc_auc": float(roc_auc_score(y, score)),
        "at_top_k_per_month": precision_recall_at_k(y, score, months),
        "per_month": per_month_ap(y, score, months),
    }
    if probability:
        result["brier"] = float(brier_score_loss(y, score))
        result["brier_base_rate_reference"] = float(brier_score_loss(y, np.full(len(y), base)))
    if threshold is not None:
        result["confusion"] = confusion_at(y, score, threshold)
    return result


def paired_bootstrap_ap_diff(y, score_a, score_b, grid_ids, n_boot=300, seed=0):
    """95% CI of AP(score_a) - AP(score_b), resampling GRIDS with replacement."""
    y = np.asarray(y)
    codes, uniques = pd.factorize(pd.Series(grid_ids))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, len(uniques), len(uniques)), minlength=len(uniques))
        w = counts[codes].astype(float)
        diffs.append(average_precision_score(y, score_a, sample_weight=w) - average_precision_score(y, score_b, sample_weight=w))
    diffs = np.array(diffs)
    return {
        "point_diff": float(average_precision_score(y, score_a) - average_precision_score(y, score_b)),
        "ci95": [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))],
        "n_bootstrap": n_boot,
    }


def assign_tiers(score, months, fractions):
    """Rank-based risk tiers within each month.

    `fractions` maps tier -> cumulative top share, e.g. {"Critical": .01, "High": .05, "Medium": .10}.
    Everything below the largest share is "Low".
    """
    score = np.asarray(score, dtype=float)
    months = np.asarray(months)
    tiers = np.full(len(score), "Low", dtype=object)
    rng = np.random.default_rng(0)
    for m in np.unique(months):
        idx = np.where(months == m)[0]
        order = idx[_order(score[idx], rng)]
        # assign from the widest tier to the narrowest so narrower tiers overwrite
        for name, frac in sorted(fractions.items(), key=lambda kv: -kv[1]):
            k = max(1, int(np.ceil(frac * len(idx))))
            tiers[order[:k]] = name
    return tiers
