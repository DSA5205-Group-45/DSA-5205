"""Minimal assert-based tests for the timing/leakage-critical pieces.

Run:  python code/test_backtest.py   (no pytest required)
"""

import numpy as np
import pandas as pd

from features import build_dataset
from run_backtest import TRAIN_END, VAL_END, max_drawdown, split_dataset


def test_max_drawdown_counts_first_day_loss():
    ret = np.array([-0.10, 0.05, -0.02])
    dd = max_drawdown(ret)
    assert abs(dd - (-0.10)) < 1e-12, f"first-day loss not counted: {dd}"


def test_split_respects_label_availability():
    ds = build_dataset().dropna(subset=["naive_excess"]).reset_index(drop=True)
    train, val, test, trainval = split_dataset(ds)
    val_start = ds.loc[ds["date"] > TRAIN_END, "date"].min()
    test_start = ds.loc[ds["date"] > VAL_END, "date"].min()
    # every label used for fitting or selection is realized by the first OOS decision
    assert train["label_end_date"].max() <= val_start
    assert val["label_end_date"].max() <= test_start
    assert trainval["label_end_date"].max() <= test_start
    # boundary rows really were dropped (2014-12-31 and 2019-12-31 labels are
    # unrealized at the first OOS decision)
    assert train["date"].max() < pd.Timestamp(TRAIN_END)
    assert val["date"].max() < pd.Timestamp(VAL_END)
    # test rows are untouched
    assert (test["date"] > VAL_END).all()


def test_naive_forecast_uses_only_realized_labels():
    ds = build_dataset()
    # at row t the naive forecast may only use labels realized by close t,
    # i.e. labels of rows <= t-2; check the construction directly
    manual = ds["fwd_excess"].expanding().mean().shift(2)
    assert np.allclose(ds["naive_excess"].dropna().to_numpy(),
                       manual.dropna().to_numpy())


if __name__ == "__main__":
    test_max_drawdown_counts_first_day_loss()
    print("[pass] max_drawdown counts first-day loss")
    test_split_respects_label_availability()
    print("[pass] split respects label availability at boundaries")
    test_naive_forecast_uses_only_realized_labels()
    print("[pass] naive forecast uses only realized labels")
    print("all tests passed")
