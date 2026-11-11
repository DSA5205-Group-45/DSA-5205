"""Regression tests for the preserved nine-strategy Ridge extension."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ridge_extension import NINE_STRATEGIES, RIDGE_STRATEGIES, MOMENTUM_STRATEGIES
from strategies import capped, pools

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(PROJECT_ROOT / "tests"))
from extension_fixture import DummyModel, make_selections, make_top300  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_preserved_sources_are_unmodified():
    expected = json.loads((PROJECT_ROOT / "config" / "preserved_sources.json").read_text())
    assert _sha256(PROJECT_ROOT / "strategies.py") == expected["strategies.py"]
    assert _sha256(PROJECT_ROOT / "src" / "pipeline.py") == expected["src/pipeline.py"]


def test_exactly_nine_strategies():
    assert len(NINE_STRATEGIES) == 9
    assert set(NINE_STRATEGIES) == set(MOMENTUM_STRATEGIES + RIDGE_STRATEGIES + ["SPY", "QQQ"])
    assert MOMENTUM_STRATEGIES == ["B1_Momentum10", "B2_Momentum30", "B3_MomentumDoubleSort10"]
    assert RIDGE_STRATEGIES == [
        "R1_RidgeWeighted10",
        "R2_RidgeFree60",
        "R3_RidgeQuota343",
        "R4_RidgeDoubleSort30",
    ]


def test_strategy_identifiers_and_weights_match_golden_fixture():
    golden = json.loads((PROJECT_ROOT / "tests" / "fixtures" / "extension_golden.json").read_text())
    selections, s60, s30 = make_selections()
    assert sorted(selections) == sorted(golden["selections"])
    assert s60 == golden["s60"]
    assert s30 == golden["s30"]
    for name, weights in selections.items():
        assert set(weights) == set(golden["selections"][name])
        for ticker, weight in weights.items():
            assert np.isclose(weight, golden["selections"][name][ticker], atol=1e-12)
        assert np.isclose(sum(weights.values()), 1.0)


def test_b3_is_momentum_top10_of_the_double_sorted_pool():
    golden = json.loads((PROJECT_ROOT / "tests" / "fixtures" / "extension_golden.json").read_text())
    top = make_top300()
    s60, s30 = pools(top)
    expected = s30.sort_values(["mom_2_12", "ticker"], ascending=[False, True]).head(10).ticker.tolist()
    assert set(golden["selections"]["B3_MomentumDoubleSort10"]) == set(expected)


def test_r1_holds_the_same_stocks_as_b1():
    golden = json.loads((PROJECT_ROOT / "tests" / "fixtures" / "extension_golden.json").read_text())
    assert set(golden["selections"]["R1_RidgeWeighted10"]) == set(golden["selections"]["B1_Momentum10"])


def test_r1_weights_respect_the_twenty_percent_cap():
    golden = json.loads((PROJECT_ROOT / "tests" / "fixtures" / "extension_golden.json").read_text())
    weights = golden["selections"]["R1_RidgeWeighted10"]
    assert np.isclose(sum(weights.values()), 1.0)
    assert max(weights.values()) <= 0.2 + 1e-9


def test_capped_handles_nonpositive_scores():
    weights = capped([0.0] * 10)
    assert np.isclose(weights.sum(), 1.0)
    assert np.allclose(weights, 0.1)
    capped_weights = capped([1.0] + [0.0] * 9)
    assert np.isclose(capped_weights.sum(), 1.0)
    assert capped_weights.max() <= 0.2 + 1e-9


def test_r3_quota_is_three_four_three():
    golden = json.loads((PROJECT_ROOT / "tests" / "fixtures" / "extension_golden.json").read_text())
    assert len(golden["selections"]["R3_RidgeQuota343"]) == 10


def test_simulate_produces_the_seven_investable_sleeves():
    from strategies import simulate

    selections, _, _ = make_selections()
    dates = pd.to_datetime(["2022-01-31", "2022-02-28"])
    by_date = {date: selections for date in dates}
    all_tickers = {ticker for weights in selections.values() for ticker in weights}
    lookup = {(date, ticker): 0.01 for date in dates for ticker in all_tickers}
    returns, trades, holdings = simulate(by_date, lookup, 20)
    assert set(returns.strategy) == set(MOMENTUM_STRATEGIES + RIDGE_STRATEGIES)
    assert len(returns) == 2 * 7
    assert (returns.turnover_two_way >= 0).all()


@pytest.mark.parametrize(
    "strategy,expected",
    [
        ("R1_RidgeWeighted10", 0.8788683515930791),
        ("B1_Momentum10", 0.7312851953949475),
        ("B3_MomentumDoubleSort10", 0.5151040491936363),
        ("R4_RidgeDoubleSort30", 0.4421475103782435),
        ("R2_RidgeFree60", 0.37667973696741464),
        ("B2_Momentum30", 0.3140299452991602),
        ("R3_RidgeQuota343", 0.22528555318602317),
        ("QQQ", 0.1480868553515977),
        ("SPY", 0.12717244601778943),
    ],
)
def test_real_extension_reproduces_reference_summary(strategy, expected):
    path = PROJECT_ROOT / "results" / "extensions" / "strategy_summary.csv"
    if not path.exists():
        pytest.skip("Extension results not generated in this checkout")
    summary = pd.read_csv(path).set_index("strategy")
    assert strategy in summary.index
    assert np.isclose(summary.loc[strategy, "annualized_return"], expected, atol=1e-6)


def test_extension_outputs_cover_nine_strategies():
    path = PROJECT_ROOT / "results" / "extensions" / "strategy_summary.csv"
    if not path.exists():
        pytest.skip("Extension results not generated in this checkout")
    summary = pd.read_csv(path)
    assert len(summary) == 9
    assert set(summary.strategy) == set(NINE_STRATEGIES)
