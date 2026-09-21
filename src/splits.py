"""Chronological split by TARGET month (the month being predicted)."""

import pandas as pd

from config import TRAIN_END_MONTH, VAL_YEAR, TEST_YEAR


def assign_split(df):
    """Return a Series with values train / val / test / unlabeled / none.

    Rows are assigned by the month their label refers to (t+1), not by the
    month the features were computed, so no train label lies in val/test.
    """
    target = pd.to_datetime(df["target_month"])
    train_end = pd.Period(TRAIN_END_MONTH, "M").to_timestamp()
    split = pd.Series("none", index=df.index)
    split[target <= train_end] = "train"
    split[target.dt.year == VAL_YEAR] = "val"
    split[target.dt.year == TEST_YEAR] = "test"
    split[df["hotspot"].isna()] = "unlabeled"
    return split
