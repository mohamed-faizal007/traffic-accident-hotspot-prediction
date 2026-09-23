"""Alternate quantile-based LOW/MEDIUM/HIGH risk tiers (DA1 proposal item:
"LOW/MEDIUM/HIGH risk tiers via quantile thresholds").

This is shown ALONGSIDE, and never replaces, the frozen classifier's
rank-based 4-tier system (Critical/High/Medium/Low, src/evaluation.py::assign_tiers,
results/frozen_config.json["risk_tiers"]). It only reads the frozen model's
already-computed scores from outputs/test_predictions_2025.parquet -- it
never re-runs the model, touches the lock file, or changes any frozen
artifact.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import OUTPUTS_DIR  # noqa: E402

QUANTILE_CUTS = (1 / 3, 2 / 3)  # terciles
TIER_LABELS = ("LOW", "MEDIUM", "HIGH")


def assign_quantile_tiers(score, months, cuts=QUANTILE_CUTS):
    """LOW/MEDIUM/HIGH tiers from within-month score terciles.

    Computed per month (same convention as the existing rank-based tiers in
    src/evaluation.py::assign_tiers) so the categorisation reflects each
    month's own score distribution rather than pooling across months with
    very different base rates.
    """
    score = np.asarray(score, dtype=float)
    months = np.asarray(months)
    tiers = np.empty(len(score), dtype=object)
    for m in np.unique(months):
        idx = np.where(months == m)[0]
        q_low, q_high = np.quantile(score[idx], cuts)
        tiers[idx] = np.select(
            [score[idx] <= q_low, score[idx] <= q_high],
            [TIER_LABELS[0], TIER_LABELS[1]],
            default=TIER_LABELS[2],
        )
    return tiers


def build_quantile_tier_table(predictions_path=None):
    """Read-only: load the frozen model's test-set predictions and attach the
    alternate quantile tier as a new column, without touching the source file
    or any frozen artifact. Returns a DataFrame; does not write anything."""
    path = predictions_path or (OUTPUTS_DIR / "test_predictions_2025.parquet")
    df = pd.read_parquet(path)
    out = df.copy()
    out["quantile_tier"] = assign_quantile_tiers(out["score"].to_numpy(), out["target_month"].to_numpy())
    return out


def main():
    table = build_quantile_tier_table()
    dest_dir = Path(__file__).resolve().parent.parent.parent / "results" / "regression"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "quantile_tiers_2025.parquet"
    table.to_parquet(dest, index=False)
    print(f"Wrote {len(table):,} rows with rank-tier + quantile-tier side by side -> {dest}")
    print(table["risk_tier"].value_counts())
    print(table["quantile_tier"].value_counts())


if __name__ == "__main__":
    main()
