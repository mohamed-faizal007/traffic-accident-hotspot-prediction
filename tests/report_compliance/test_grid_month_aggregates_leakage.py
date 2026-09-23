"""Synthetic-grid tests proving grid-month aggregate features (rush-hour
share, weekend share, road/weather shares) at month t use only collisions
from months <= t -- same method as tests/test_features_leakage.py."""

import numpy as np
import pandas as pd
import pytest

from report_compliance.grid_month_aggregates import (
    build_aggregate_features,
    FLAG_COLUMNS,
    SHARE_COLUMNS,
    NEVER_SEEN_SENTINEL,
)

MONTHS = pd.date_range("2021-01-01", periods=4, freq="MS")


def make_skeleton(grid_ids):
    return pd.DataFrame(
        [{"grid_id": g, "year_month": m} for g in grid_ids for m in MONTHS]
    )


def collision(grid_id, month_idx, time, day_of_week, weather=1, surface=1, road_type=3, speed=30):
    return dict(
        grid_id=grid_id,
        year_month=MONTHS[month_idx],
        time=time,
        day_of_week=day_of_week,
        weather_conditions=weather,
        road_surface_conditions=surface,
        road_type=road_type,
        speed_limit=speed,
    )


def base_collisions():
    """Grid A: one collision in month 0, one in month 1, none in month 2,
    one in month 3. Grid B: no collisions ever (zero-history sentinel case)."""
    return [
        collision("A", 0, "12:00", day_of_week=2),                                    # plain weekday midday
        collision("A", 1, "08:00", day_of_week=1, weather=2, surface=2, road_type=6, speed=50),  # everything "on"
        collision("A", 3, "02:00", day_of_week=2, road_type=6),                        # night, single carriageway
    ]


def test_aggregates_ignore_future_months():
    """Changing month 3's collisions must not change months 0-2's shares."""
    skeleton = make_skeleton(["A", "B"])
    base = build_aggregate_features(pd.DataFrame(base_collisions()), skeleton)

    perturbed_rows = base_collisions()
    # wildly different month-3 collision: change every flag
    perturbed_rows[-1] = collision("A", 3, "23:00", day_of_week=1, weather=9, surface=9, road_type=1, speed=70)
    perturbed = build_aggregate_features(pd.DataFrame(perturbed_rows), skeleton)

    early = base["year_month"] < MONTHS[3]
    a = base.loc[early, SHARE_COLUMNS].to_numpy()
    b = perturbed.loc[early, SHARE_COLUMNS].to_numpy()
    np.testing.assert_allclose(a, b, err_msg="aggregate shares at months <3 depend on month-3 data")


def test_aggregates_do_depend_on_current_month():
    """Sanity check: changing month 1's OWN collision does change month 1's shares."""
    skeleton = make_skeleton(["A", "B"])
    base = build_aggregate_features(pd.DataFrame(base_collisions()), skeleton)

    changed_rows = base_collisions()
    changed_rows[1] = collision("A", 1, "12:00", day_of_week=2)  # now plain instead of "everything on"
    changed = build_aggregate_features(pd.DataFrame(changed_rows), skeleton)

    m1_base = base[(base.grid_id == "A") & (base.year_month == MONTHS[1])][SHARE_COLUMNS].to_numpy()
    m1_changed = changed[(changed.grid_id == "A") & (changed.year_month == MONTHS[1])][SHARE_COLUMNS].to_numpy()
    assert not np.allclose(m1_base, m1_changed)


def test_hand_computed_values():
    skeleton = make_skeleton(["A"])
    df = build_aggregate_features(pd.DataFrame(base_collisions()), skeleton)
    df = df.sort_values("year_month").reset_index(drop=True)

    def row(i):
        return df.iloc[i][SHARE_COLUMNS].to_dict()

    # month 0: single plain weekday-midday collision -> everything 0/1 = 0.0
    r0 = row(0)
    for col in SHARE_COLUMNS:
        assert r0[col] == pytest.approx(0.0), col

    # month 1: cumulative totals = 2 collisions; the month-1 collision trips
    # rush(1), weekend(1), adverse(1), wet(1), single_carriageway(1), high_speed(1); night stays 0
    r1 = row(1)
    assert r1["historical_rush_hour_share"] == pytest.approx(0.5)
    assert r1["historical_weekend_share"] == pytest.approx(0.5)
    assert r1["historical_night_share"] == pytest.approx(0.0)
    assert r1["historical_adverse_weather_share"] == pytest.approx(0.5)
    assert r1["historical_wet_road_share"] == pytest.approx(0.5)
    assert r1["historical_single_carriageway_share"] == pytest.approx(0.5)
    assert r1["historical_high_speed_share"] == pytest.approx(0.5)

    # month 2: no NEW collisions -> cumulative counts carry forward unchanged from month 1
    r2 = row(2)
    assert r2 == pytest.approx(r1)

    # month 3: cumulative totals = 3; a new night, single-carriageway, weekday,
    # fine-weather, dry, low-speed collision is added
    r3 = row(3)
    assert r3["historical_rush_hour_share"] == pytest.approx(1 / 3)
    assert r3["historical_weekend_share"] == pytest.approx(1 / 3)
    assert r3["historical_night_share"] == pytest.approx(1 / 3)
    assert r3["historical_adverse_weather_share"] == pytest.approx(1 / 3)
    assert r3["historical_wet_road_share"] == pytest.approx(1 / 3)
    assert r3["historical_single_carriageway_share"] == pytest.approx(2 / 3)
    assert r3["historical_high_speed_share"] == pytest.approx(1 / 3)


def test_never_seen_sentinel_is_not_zero_or_valid_share():
    """A grid with zero collisions ever recorded must get the sentinel, not 0.0
    (which would look like a real, informative 0% share)."""
    skeleton = make_skeleton(["A", "B"])
    df = build_aggregate_features(pd.DataFrame(base_collisions()), skeleton)
    b_rows = df[df.grid_id == "B"]
    for col in SHARE_COLUMNS:
        assert (b_rows[col] == NEVER_SEEN_SENTINEL).all(), col
        assert NEVER_SEEN_SENTINEL < 0  # outside the valid [0, 1] share range


def test_flag_columns_and_share_columns_correspond_one_to_one():
    assert len(FLAG_COLUMNS) == len(SHARE_COLUMNS) == 7
    assert len(set(SHARE_COLUMNS)) == len(SHARE_COLUMNS)  # no accidental name collisions
