"""Deterministic fixture for the nine-strategy extension regression test.

Kept in a helper module so the golden snapshot and the test are generated from
exactly the same input construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "mom_21",
    "mom_63",
    "mom_126",
    "mom_2_12",
    "relative_mom_63",
    "vol_21",
    "vol_63",
    "beta_252",
    "drawdown_63",
    "ma_ratio_50",
    "ma_ratio_200",
    "rsi_14",
    "ln_adv_20",
    "volume_ratio_21",
    "range_hl",
]


def make_top300(n: int = 300, seed: int = 2024) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "ticker": [f"T{i:03d}" for i in range(n)],
            "mom_2_12": np.linspace(2.5, -2.5, n) + rng.normal(0, 0.02, n),
            "vol_63": np.tile(np.linspace(0.10, 0.90, 100), 3),
        }
    )
    for feature in FEATURES:
        if feature not in frame.columns:
            frame[feature] = rng.normal(0.0, 0.2, n)
    frame["close"] = 50.0
    frame["adv_20"] = 1e8
    return frame


class DummyModel:
    """Deterministic stand-in: predicted return equals momentum."""

    def predict(self, X):
        return X["mom_2_12"].to_numpy(dtype=float)


def make_selections():
    from strategies import pools, targets_for_date

    top = make_top300()
    s60, s30 = pools(top)
    selections = targets_for_date(top, s60, s30, DummyModel())
    ordered = {name: dict(sorted(weights.items())) for name, weights in sorted(selections.items())}
    return ordered, sorted(s60.ticker.tolist()), sorted(s30.ticker.tolist())
