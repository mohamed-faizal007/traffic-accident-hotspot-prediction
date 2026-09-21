"""Synthetic-grid tests proving features at month t use only months <= t."""

import numpy as np
import pandas as pd
import pytest

from features import build_feature_table, FEATURE_COLUMNS
from splits import assign_split

N_MONTHS = 36  # 2021-01 .. 2023-12


def make_monthly(counts_by_grid, start="2021-01-01", casualties=None):
    rows = []
    months = pd.date_range(start, periods=len(next(iter(counts_by_grid.values()))), freq="MS")
    for gi, (gid, counts) in enumerate(counts_by_grid.items()):
        for m, c in zip(months, counts):
            rows.append(
                dict(grid_id=gid, year_month=m, collision_count=c, total_casualties=c + 1,
                     grid_latitude=51.0 + gi, grid_longitude=-1.0 - gi)
            )
    return pd.DataFrame(rows)


@pytest.fixture
def series():
    rng = np.random.RandomState(0)
    return {"A": rng.poisson(1.0, N_MONTHS), "B": rng.poisson(0.3, N_MONTHS)}


def test_features_ignore_future_months(series):
    """Changing any collision count AFTER month t must not change month t's features."""
    base = build_feature_table(make_monthly(series), warmup=0)
    for t in range(0, N_MONTHS - 1, 5):
        perturbed = {g: c.copy() for g, c in series.items()}
        for g in perturbed:
            perturbed[g][t + 1:] += 7          # huge change to the future
        other = build_feature_table(make_monthly(perturbed), warmup=0)
        a = base[base.month_index == t].sort_values("grid_id")[FEATURE_COLUMNS].to_numpy()
        b = other[other.month_index == t].sort_values("grid_id")[FEATURE_COLUMNS].to_numpy()
        np.testing.assert_allclose(a, b, err_msg=f"features at month {t} depend on the future")


def test_features_do_depend_on_current_month(series):
    """Sanity check for the test above: changing month t itself DOES change features."""
    base = build_feature_table(make_monthly(series), warmup=0)
    changed = {g: c.copy() for g, c in series.items()}
    changed["A"][10] += 5
    other = build_feature_table(make_monthly(changed), warmup=0)
    a = base[(base.month_index == 10) & (base.grid_id == "A")][FEATURE_COLUMNS].to_numpy()
    b = other[(other.month_index == 10) & (other.grid_id == "A")][FEATURE_COLUMNS].to_numpy()
    assert not np.allclose(a, b)


def test_target_is_next_month(series):
    df = build_feature_table(make_monthly(series), warmup=0, hotspot_min=2)
    for gid, counts in series.items():
        g = df[df.grid_id == gid].sort_values("month_index")
        expected = (counts[1:] >= 2).astype(float)
        np.testing.assert_array_equal(g["hotspot"].to_numpy()[:-1], expected)
        assert np.isnan(g["hotspot"].to_numpy()[-1])        # last month has no label
        np.testing.assert_array_equal(g["next_month_collisions"].to_numpy()[:-1], counts[1:])


def test_hand_computed_values():
    counts = [0, 1, 0, 3, 0, 0, 2, 0]
    df = build_feature_table(make_monthly({"A": counts}), warmup=0)
    row = df[df.month_index == 6].iloc[0]          # month t=6 (count 2)
    assert row.collisions_lag_0 == 2 and row.collisions_lag_1 == 0 and row.collisions_lag_2 == 0
    assert row.collisions_rolling_sum_3 == 2       # months 4,5,6
    assert row.collisions_last_6_months == 6       # months 1..6: 1+0+3+0+0+2
    assert row.historical_total_collisions == 6    # months 0..6
    assert row.historical_max_collisions == 3
    assert row.months_since_last_collision == 0    # collision in month t itself
    assert row.hotspot == 0                        # month 7 has 0 collisions
    row = df[df.month_index == 5].iloc[0]
    assert row.months_since_last_collision == 2    # last collision in month 3
    assert row.historical_hotspot_frequency == pytest.approx(1 / 6)   # only month 3 had >=2


def test_delay_reproduces_old_alignment():
    """delay=1: month t's own count must be invisible; label is still month t+1."""
    counts = [0, 1, 0, 3, 0, 0, 2, 0]
    df = build_feature_table(make_monthly({"A": counts}), delay=1, warmup=0)
    row = df[df.month_index == 6].iloc[0]
    assert row.collisions_lag_0 == 0               # = count of month 5
    assert row.collisions_lag_1 == 0 and row.collisions_lag_2 == 3
    assert row.next_month_collisions == 0          # label unchanged


def test_never_seen_sentinel_is_not_zero():
    df = build_feature_table(make_monthly({"A": [0, 0, 0, 0, 1, 0]}), warmup=0)
    row = df[df.month_index == 2].iloc[0]
    assert row.months_since_last_collision > 0     # 0 would mean "collision this month"


def test_split_by_target_month():
    monthly = make_monthly({"A": np.ones(60, dtype=int)})
    df = build_feature_table(monthly, warmup=0)
    df["split"] = assign_split(df)
    tm = pd.to_datetime(df["target_month"])
    assert tm[df.split == "train"].max() == pd.Timestamp("2023-12-01")
    assert set(tm[df.split == "val"].dt.year) == {2024}
    assert set(tm[df.split == "test"].dt.year) == {2025}
    # Dec-2024 features (label = Jan 2025) belong to TEST, not train
    dec24 = df[(df.year_month == "2024-12-01")]
    assert (dec24.split == "test").all()
    assert (df[df.year_month == "2025-12-01"].split == "unlabeled").all()


def test_active_grids_use_training_period_only():
    from temporal_dataset import select_active_grids
    months = pd.to_datetime(["2022-05-01", "2024-01-01", "2024-02-01", "2025-01-01", "2025-02-01", "2025-03-01"])
    rows = []
    # grid "late": 3 collisions but only after the training period -> NOT active
    for m in months[3:]:
        rows.append(dict(grid_id="late", year_month=m, collision_index=len(rows), datetime=m))
    # grid "early": 3 collisions in 2021-2023 -> active
    for m in pd.to_datetime(["2021-02-01", "2022-06-01", "2023-11-01"]):
        rows.append(dict(grid_id="early", year_month=m, collision_index=len(rows), datetime=m))
    active = select_active_grids(pd.DataFrame(rows))
    assert list(active) == ["early"]
