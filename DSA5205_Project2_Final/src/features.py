"""Feature registry and preprocessing contract.

Feature *construction* lives in the preserved :mod:`src.pipeline` and is not
re-implemented here.  This module centralises the feature list, human-readable
descriptions and the leakage-safe preprocessing helper used by the linear
models, so there is exactly one definition of the model input contract.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .pipeline import FEATURES

FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "mom_21": "21-session total return",
    "mom_63": "63-session total return",
    "mom_126": "126-session total return",
    "mom_2_12": "12-month return skipping the most recent month (actual monthly observations)",
    "relative_mom_63": "63-session return minus SPY 63-session return",
    "vol_21": "annualised 21-session realised volatility",
    "vol_63": "annualised 63-session realised volatility",
    "beta_252": "252-session rolling beta to SPY",
    "drawdown_63": "current price relative to its 63-session high",
    "ma_ratio_50": "price relative to its 50-session moving average",
    "ma_ratio_200": "price relative to its 200-session moving average",
    "rsi_14": "14-session relative strength index",
    "ln_adv_20": "log of 20-session average daily dollar volume",
    "volume_ratio_21": "volume relative to its 21-session average",
    "range_hl": "daily high-low range scaled by close",
}

SCREENING_FEATURES = {"mom_2_12", "vol_63", "ln_adv_20"}
FORWARD_FILLED_FEATURES = [f for f in FEATURES if f not in SCREENING_FEATURES]


def feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the model design matrix in the canonical column order."""
    missing = [column for column in FEATURES if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    matrix = frame.loc[:, FEATURES].astype(float)
    if not np.isfinite(matrix.to_numpy()).all():
        raise ValueError("Non-finite values in the feature matrix")
    return matrix


def assert_feature_contract(frame: pd.DataFrame) -> None:
    """Validate the feature contract used by all four models."""
    if len(FEATURES) != 15:
        raise AssertionError("The model contract expects exactly 15 features")
    feature_matrix(frame)


def scaled_pipeline(estimator) -> Pipeline:
    """Leakage-safe preprocessing for the linear models.

    ``StandardScaler`` is fitted by ``Pipeline.fit`` on whatever rows are passed
    in.  The pipeline therefore never sees validation or test rows during
    training, provided callers fit on the training slice only.
    """
    return Pipeline([("scale", StandardScaler()), ("model", estimator)])


def describe_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": FEATURES,
            "description": [FEATURE_DESCRIPTIONS[f] for f in FEATURES],
            "screening_feature": [f in SCREENING_FEATURES for f in FEATURES],
            "forward_filled_in_pipeline": [f in FORWARD_FILLED_FEATURES for f in FEATURES],
        }
    )
