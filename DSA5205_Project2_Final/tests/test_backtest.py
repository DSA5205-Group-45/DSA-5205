"""Backtest invariants: Top-10, equal weights, costs, turnover, execution."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.backtest import (
    assert_execution_ordering,
    portfolio_backtest,
    reconcile_holdings,
    run_single_strategy_backtest,
)
from src.pipeline import attach_labels


def test_top10_and_equal_weight_are_preserved(synthetic_predictions):
    returns, trades, holdings = run_single_strategy_backtest(synthetic_predictions, "ridge")
    first_date = synthetic_predictions.decision_date.min()
    first = holdings.loc[holdings.decision_date == first_date]
    assert len(first) == 10
    assert np.allclose(first.target_weight, 0.1)
    expected = (
        synthetic_predictions.loc[synthetic_predictions.decision_date == first_date]
        .sort_values(["prediction", "ticker"], ascending=[False, True])
        .head(10)
        .ticker.tolist()
    )
    assert set(first.ticker) == set(expected)
    assert np.isclose(first.target_weight.sum(), 1.0)


def test_original_three_strategies_still_exist(synthetic_predictions):
    returns, _, _ = portfolio_backtest(synthetic_predictions, "monthly", 20, top_n=10)
    assert set(returns.strategy) == {"model_top10", "momentum_top10", "equal_weight_30", "SPY", "QQQ"}


def _stable_predictions(synthetic_predictions):
    frame = synthetic_predictions.copy()
    frame["prediction"] = -frame["ticker"].str[1:].astype(int).astype(float)
    return frame


def test_first_period_cost_and_drifted_turnover(synthetic_predictions):
    predictions = _stable_predictions(synthetic_predictions)
    returns, _, _ = run_single_strategy_backtest(predictions, "ridge")
    model = returns.loc[returns.strategy == "ridge"].sort_values("decision_date").reset_index(drop=True)
    # Fully invested from cash: turnover is 1.0 and cost is 20 bps.
    assert np.isclose(model.loc[0, "turnover_two_way"], 1.0)
    assert np.isclose(model.loc[0, "cost"], 0.002)
    # Second period: only drift-driven trading, never a full liquidation.
    assert model.loc[1, "turnover_two_way"] < 1.0
    assert np.isclose(model.loc[1, "cost"], model.loc[1, "turnover_two_way"] * 20 / 10000)


def test_turnover_matches_manual_calculation(synthetic_predictions):
    returns, _, _ = run_single_strategy_backtest(synthetic_predictions, "ridge")
    model = returns.loc[returns.strategy == "ridge"].sort_values("decision_date").reset_index(drop=True)
    dates = sorted(synthetic_predictions.decision_date.unique())
    first = synthetic_predictions.loc[synthetic_predictions.decision_date == dates[0]].sort_values(
        ["prediction", "ticker"], ascending=[False, True]
    ).head(10)
    target_1 = {t: 0.1 for t in first.ticker}
    realized_1 = dict(zip(first.ticker, first.target_forward_return))
    grown = {t: w * (1 + realized_1[t]) for t, w in target_1.items()}
    denominator = sum(grown.values())
    pre = {t: value / denominator for t, value in grown.items()}
    second = synthetic_predictions.loc[synthetic_predictions.decision_date == dates[1]].sort_values(
        ["prediction", "ticker"], ascending=[False, True]
    ).head(10)
    target_2 = {t: 0.1 for t in second.ticker}
    manual = sum(abs(target_2.get(t, 0) - pre.get(t, 0)) for t in set(pre) | set(target_2))
    assert np.isclose(model.loc[1, "turnover_two_way"], manual)


def test_monthly_returns_reconcile_with_stock_level_returns(synthetic_predictions):
    returns, _, holdings = run_single_strategy_backtest(synthetic_predictions, "ridge")
    assert reconcile_holdings(returns, holdings) < 1e-10


def test_benchmarks_are_aligned_and_charged_once(synthetic_predictions):
    returns, _, _ = run_single_strategy_backtest(synthetic_predictions, "ridge")
    model_dates = set(returns.loc[returns.strategy == "ridge", "decision_date"])
    for benchmark in ("SPY", "QQQ"):
        block = returns.loc[returns.strategy == benchmark].sort_values("decision_date")
        assert set(block.decision_date) == model_dates
        assert np.isclose(block.iloc[0]["cost"], 0.002)
        assert np.allclose(block.iloc[1:]["cost"], 0.0)
        assert np.allclose(block.iloc[1:]["turnover_two_way"], 0.0)


def test_execution_strictly_after_decision(synthetic_predictions):
    assert assert_execution_ordering(synthetic_predictions)


def _tiny_panel():
    dates = pd.bdate_range("2019-12-02", "2020-03-02")
    dates = dates[dates != pd.Timestamp("2020-01-01")]
    rows = []
    for ticker_index, ticker in enumerate(["AAA", "BBB", "SPY", "QQQ"]):
        for day, date in enumerate(dates):
            base = 100.0 + 10 * ticker_index + day
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": base,
                    "high": base + 1,
                    "low": base - 1,
                    "close": base + 0.5,
                    "adj_close": base * 1.01,
                    "volume": 1_000_000,
                    "dividends": 0.0,
                    "stock_splits": 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_labels_use_next_open_to_next_rebalance_open():
    panel = _tiny_panel()
    cfg = {
        "research_start": "2019-12-01",
        "research_end": "2019-12-31",
        "rebalance_frequency": "monthly",
        "train_end_exclusive": "2020-01-01",
        "validation_end_exclusive": "2022-01-01",
    }
    screened = pd.DataFrame(
        {"date": [pd.Timestamp("2019-12-31")] * 2, "ticker": ["AAA", "BBB"]}
    )
    labeled = attach_labels(screened, panel, cfg)
    assert (labeled.execution_date == pd.Timestamp("2020-01-02")).all()
    assert (labeled.label_end_date == pd.Timestamp("2020-02-03")).all()
    assert (labeled.decision_date < labeled.execution_date).all()

    for _, row in labeled.iterrows():
        expected = (
            panel.loc[(panel.ticker == row.ticker) & (panel.date == row.label_end_date), "open"].iloc[0]
            * panel.loc[(panel.ticker == row.ticker) & (panel.date == row.label_end_date), "adj_close"].iloc[0]
            / panel.loc[(panel.ticker == row.ticker) & (panel.date == row.label_end_date), "close"].iloc[0]
        ) / (
            panel.loc[(panel.ticker == row.ticker) & (panel.date == row.execution_date), "open"].iloc[0]
            * panel.loc[(panel.ticker == row.ticker) & (panel.date == row.execution_date), "adj_close"].iloc[0]
            / panel.loc[(panel.ticker == row.ticker) & (panel.date == row.execution_date), "close"].iloc[0]
        ) - 1.0
        assert np.isclose(row["target_forward_return"], expected)


def _function_hash(path, name):
    import ast
    import hashlib

    source = Path(path).read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest()
    raise KeyError(name)


def test_preserved_backtest_functions_are_unchanged():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    expected = json.loads((root / "config" / "preserved_sources.json").read_text())["backtest_functions"]
    for name, digest in expected.items():
        assert _function_hash(root / "src" / "backtest.py", name) == digest, name
