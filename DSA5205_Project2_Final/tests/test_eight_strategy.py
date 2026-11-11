"""Eight-strategy OOS reconciliation: alignment, B3 rule, metrics and figures."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EIGHT = PROJECT_ROOT / "results" / "oos" / "eight_strategy"
EXPECTED = [
    "ridge",
    "lasso",
    "rf",
    "xgb",
    "B1_Momentum10",
    "B3_MomentumDoubleSort10",
    "SPY",
    "QQQ",
]


def _skip_if_missing():
    if not (EIGHT / "metrics.csv").exists():
        pytest.skip("Eight-strategy comparison not generated in this checkout")


def test_eight_strategies_present():
    _skip_if_missing()
    metrics = pd.read_csv(EIGHT / "metrics.csv")
    assert set(metrics.strategy) == set(EXPECTED)
    assert len(metrics) == 8


def test_uniform_periods_and_dates():
    _skip_if_missing()
    panel = pd.read_csv(EIGHT / "monthly_returns.csv", parse_dates=["decision_date"])
    counts = panel.groupby("strategy").size()
    assert counts.nunique() == 1
    assert int(counts.iloc[0]) == 48
    dates = panel.groupby("strategy").decision_date.apply(lambda s: tuple(sorted(s)))
    assert dates.nunique() == 1
    assert panel.decision_date.min() == pd.Timestamp("2022-01-31")
    assert panel.decision_date.max() == pd.Timestamp("2025-12-31")


def test_alignment_checks_were_run_and_passed():
    _skip_if_missing()
    checks = json.loads((EIGHT / "alignment_checks.json").read_text())
    assert checks["decision_dates_identical"] is True
    assert checks["benchmarks_identical"] is True
    assert checks["b3_rule_verified"] is True
    assert checks["b3_mismatched_dates"] == []
    assert checks["periods_per_strategy"] == 48
    assert checks["benchmark_max_abs_difference"]["SPY"] <= 1e-12
    assert checks["benchmark_max_abs_difference"]["QQQ"] <= 1e-12


def test_b3_is_momentum_top10_of_the_main_pool():
    _skip_if_missing()
    predictions = pd.read_csv(PROJECT_ROOT / "results/oos/oos_predictions.csv", parse_dates=["decision_date"])
    holdings = pd.read_csv(PROJECT_ROOT / "results/extensions/holdings.csv", parse_dates=["decision_date"])
    ridge = predictions.loc[predictions.model == "ridge"]
    b3 = holdings.loc[holdings.strategy == "B3_MomentumDoubleSort10"]
    for date, block in ridge.groupby("decision_date"):
        expected = set(block.sort_values(["mom_2_12", "ticker"], ascending=[False, True]).head(10).ticker)
        observed = set(b3.loc[b3.decision_date == date, "ticker"])
        assert expected == observed, date


def test_metrics_columns_and_primary_flag():
    _skip_if_missing()
    metrics = pd.read_csv(EIGHT / "metrics.csv").set_index("strategy")
    for column in (
        "annualized_return",
        "annualized_return_gross",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "total_return_net",
        "average_turnover",
        "total_transaction_cost",
        "beta_to_benchmark",
        "alpha_annualized",
    ):
        assert column in metrics.columns, column
    assert bool(metrics.loc["ridge", "is_primary"]) is True
    assert metrics.loc["ridge", "role"] == "ML model (primary)"
    assert (metrics.loc[["B1_Momentum10", "B3_MomentumDoubleSort10"], "role"] == "Momentum baseline").all()
    assert (metrics.loc[["SPY", "QQQ"], "role"] == "Investable benchmark").all()


def test_reference_values_match_saved_outputs():
    _skip_if_missing()
    metrics = pd.read_csv(EIGHT / "metrics.csv").set_index("strategy")
    assert np.isclose(metrics.loc["ridge", "annualized_return"], 0.430430, atol=1e-5)
    assert np.isclose(metrics.loc["B1_Momentum10", "annualized_return"], 0.731285, atol=1e-5)
    assert np.isclose(metrics.loc["B3_MomentumDoubleSort10", "annualized_return"], 0.515104, atol=1e-5)
    assert np.isclose(metrics.loc["SPY", "annualized_return"], 0.127172, atol=1e-5)


def test_six_required_figures_exist():
    _skip_if_missing()
    figures = PROJECT_ROOT / "results" / "oos" / "figures"
    for number in range(1, 7):
        matches = list(figures.glob(f"fig{number}_*.png"))
        assert len(matches) == 1, number
        assert matches[0].stat().st_size > 20_000


def test_annual_returns_cover_four_years():
    _skip_if_missing()
    annual = pd.read_csv(EIGHT / "annual_returns.csv", index_col=0)
    assert list(annual.index) == [2022, 2023, 2024, 2025]
    assert set(annual.columns) == set(EXPECTED)
    assert np.isfinite(annual.to_numpy()).all()
