import numpy as np
import pandas as pd
import pytest

from report_compliance.quantile_tiers import assign_quantile_tiers, build_quantile_tier_table
from config import OUTPUTS_DIR


def test_terciles_split_roughly_evenly_within_a_month():
    rng = np.random.default_rng(0)
    score = rng.random(3000)
    months = np.full(3000, "2025-01")
    tiers = assign_quantile_tiers(score, months)
    counts = pd.Series(tiers).value_counts()
    for label in ("LOW", "MEDIUM", "HIGH"):
        assert abs(counts[label] - 1000) < 60  # within 6% of an even tercile split


def test_tiers_computed_independently_per_month():
    """A score that's high relative to its OWN month's distribution should be
    HIGH even if it would be LOW relative to a different month's distribution."""
    score = np.concatenate([np.linspace(0, 1, 300), np.linspace(100, 101, 300)])
    months = np.array(["m1"] * 300 + ["m2"] * 300)
    tiers = assign_quantile_tiers(score, months)
    # top third of month 1 (values near 1.0) should be HIGH, not LOW just
    # because month 2's absolute scores are much larger
    m1_top = tiers[:300][np.argsort(score[:300])[-50:]]
    assert (m1_top == "HIGH").all()


def test_ordering_is_monotonic_with_score():
    score = np.array([0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0, 0.95, 0.85, 0.6])
    months = np.zeros(len(score), dtype=int)
    tiers = assign_quantile_tiers(score, months)
    order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    ranked = sorted(zip(score, tiers))
    ranks = [order[t] for _, t in ranked]
    assert ranks == sorted(ranks)  # tier rank never decreases as score increases


@pytest.mark.skipif(
    not (OUTPUTS_DIR / "test_predictions_2025.parquet").exists(),
    reason="frozen classifier test predictions not present in this checkout",
)
def test_build_quantile_tier_table_does_not_mutate_source_file():
    src = OUTPUTS_DIR / "test_predictions_2025.parquet"
    before = src.read_bytes()
    table = build_quantile_tier_table()
    after = src.read_bytes()
    assert before == after, "reading frozen predictions must never modify the source file"
    assert "risk_tier" in table.columns          # existing frozen rank-based tier untouched
    assert "quantile_tier" in table.columns      # new alternate tier added, not replacing
    assert set(table["quantile_tier"].unique()) <= {"LOW", "MEDIUM", "HIGH"}
