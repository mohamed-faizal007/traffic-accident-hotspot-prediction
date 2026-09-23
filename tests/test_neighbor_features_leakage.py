"""Synthetic-grid tests proving neighbor-cell features (Moore-8 adjacency) at
month t use only neighbor collision data from months <= t -- same method as
tests/test_features_leakage.py and
tests/report_compliance/test_grid_month_aggregates_leakage.py.

Synthetic layout (grid_ix, grid_iy):
  C  = (0, 0)   -- the cell under test
  N1 = (1, 0)   -- east of C, one of the 8 Moore neighbor positions
  N2 = (0, 1)   -- north of C, another Moore neighbor position
  I  = (10, 10) -- far away, isolated: none of its 8 neighbor positions is occupied

C has only 2 of its 8 possible neighbor positions occupied (N1, N2); the other
6 positions (e.g. (-1,0), (1,1), (-1,-1), ...) simply don't exist as cells.
"""

import numpy as np
import pandas as pd
import pytest

from neighbor_features import compute_neighbor_features, NEIGHBOR_FEATURES

MONTHS = pd.date_range("2021-01-01", periods=4, freq="MS")
CELLS = {"C": (0, 0), "N1": (1, 0), "N2": (0, 1), "I": (10, 10)}


def make_monthly(counts_by_cell):
    """counts_by_cell: {cell_name: [count_month0, ..., count_month3]}. Every cell in
    CELLS must be present for every month (complete lattice), even if not in counts_by_cell
    (defaults to all-zero collision history)."""
    rows = []
    for name, (ix, iy) in CELLS.items():
        counts = counts_by_cell.get(name, [0] * len(MONTHS))
        for m, c in zip(MONTHS, counts):
            rows.append(dict(grid_id=name, grid_ix=ix, grid_iy=iy, year_month=m, collision_count=c))
    return pd.DataFrame(rows)


def base_counts():
    return {"N1": [1, 0, 0, 5], "N2": [0, 2, 0, 0]}


def c_row(df, month_idx):
    return df[(df.grid_id == "C") & (df.year_month == MONTHS[month_idx])].iloc[0]


def test_neighbor_features_ignore_future_months():
    """A huge change to a neighbor's month-3 collisions must not change C's features at months 0-2."""
    base = compute_neighbor_features(make_monthly(base_counts()))

    perturbed_counts = base_counts()
    perturbed_counts["N1"] = [1, 0, 0, 500]  # month 3 only, wildly different
    perturbed = compute_neighbor_features(make_monthly(perturbed_counts))

    early = base["year_month"] < MONTHS[3]
    a = base.loc[early & (base.grid_id == "C"), NEIGHBOR_FEATURES].to_numpy()
    b = perturbed.loc[early & (perturbed.grid_id == "C"), NEIGHBOR_FEATURES].to_numpy()
    np.testing.assert_allclose(a, b, err_msg="C's neighbor features at months <3 depend on a neighbor's month-3 data")


def test_neighbor_features_do_depend_on_neighbor_current_month():
    """Changing a neighbor's OWN month-1 collision count DOES change C's month-1 features."""
    base = compute_neighbor_features(make_monthly(base_counts()))

    changed_counts = base_counts()
    changed_counts["N1"] = [1, 10, 0, 5]  # month 1 changed from 0 to 10
    changed = compute_neighbor_features(make_monthly(changed_counts))

    a = c_row(base, 1)[NEIGHBOR_FEATURES].to_numpy(dtype=float)
    b = c_row(changed, 1)[NEIGHBOR_FEATURES].to_numpy(dtype=float)
    assert not np.allclose(a, b)


def test_hand_computed_neighbor_aggregate():
    df = compute_neighbor_features(make_monthly(base_counts()))

    # N1 cumulative avg: month0 1/1=1.0, month1 1/2=0.5, month2 1/3=0.3333, month3 6/4=1.5
    # N2 cumulative avg: month0 0/1=0.0, month1 2/2=1.0, month2 2/3=0.6667, month3 2/4=0.5
    expected = {
        0: dict(mean=(1.0 + 0.0) / 2, max=1.0, count=2),
        1: dict(mean=(0.5 + 1.0) / 2, max=1.0, count=2),
        2: dict(mean=(1 / 3 + 2 / 3) / 2, max=2 / 3, count=2),
        3: dict(mean=(1.5 + 0.5) / 2, max=1.5, count=2),
    }
    for month_idx, exp in expected.items():
        row = c_row(df, month_idx)
        assert row["neighbor_avg_collisions_mean"] == pytest.approx(exp["mean"]), month_idx
        assert row["neighbor_avg_collisions_max"] == pytest.approx(exp["max"]), month_idx
        assert row["neighbor_cell_count"] == exp["count"], month_idx


def test_edge_of_grid_has_fewer_neighbors():
    """C only has 2 of its 8 possible neighbor positions occupied; the aggregate must be
    computed over exactly those 2, not padded with 6 phantom zeros (which would bias
    mean toward 0 for every edge/irregular-footprint cell)."""
    df = compute_neighbor_features(make_monthly(base_counts()))
    row = c_row(df, 1)
    assert row["neighbor_cell_count"] == 2
    padded_mean = (0.5 + 1.0 + 6 * 0.0) / 8  # what a naive "pad with 0" implementation would give
    assert row["neighbor_avg_collisions_mean"] != pytest.approx(padded_mean)
    assert row["neighbor_avg_collisions_mean"] == pytest.approx((0.5 + 1.0) / 2)


def test_isolated_cell_has_zero_neighbor_count():
    """A cell with no occupied neighbors at all gets count 0 and mean/max fall back to 0.0."""
    df = compute_neighbor_features(make_monthly(base_counts()))
    for month_idx in range(len(MONTHS)):
        row = df[(df.grid_id == "I") & (df.year_month == MONTHS[month_idx])].iloc[0]
        assert row["neighbor_cell_count"] == 0
        assert row["neighbor_avg_collisions_mean"] == 0.0
        assert row["neighbor_avg_collisions_max"] == 0.0
