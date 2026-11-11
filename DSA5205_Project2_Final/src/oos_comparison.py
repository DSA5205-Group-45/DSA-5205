"""Unified out-of-sample comparison of eight strategies.

The main experiment and the Ridge extension were produced by two different
backtest drivers.  This module reconciles them without changing any strategy
rule:

* four ML models come from the main 30-candidate backtest;
* B1 and B3 come from the preserved extension driver;
* SPY and QQQ are taken from the main engine and verified to be identical to
  the extension engine on every decision date.

The module checks date alignment, benchmark equality and the B3 selection
rule before producing the comparison tables and Figures 1-6.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from .data_pipeline import ensure_output_dirs, load_config, resolve
from .metrics import portfolio_metrics
from .utils import utc_now, write_json

ML_MODELS = ["ridge", "lasso", "rf", "xgb"]
MOMENTUM = ["B1_Momentum10", "B3_MomentumDoubleSort10"]
BENCHMARKS = ["SPY", "QQQ"]
STRATEGIES = ML_MODELS + MOMENTUM + BENCHMARKS

DISPLAY = {
    "ridge": "Ridge (primary)",
    "lasso": "LASSO",
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "B1_Momentum10": "B1 Momentum10",
    "B3_MomentumDoubleSort10": "B3 DoubleSort10",
    "SPY": "SPY buy-and-hold",
    "QQQ": "QQQ buy-and-hold",
}
SHORT = {
    "ridge": "Ridge",
    "lasso": "LASSO",
    "rf": "RF",
    "xgb": "XGB",
    "B1_Momentum10": "B1",
    "B3_MomentumDoubleSort10": "B3",
    "SPY": "SPY",
    "QQQ": "QQQ",
}
ROLE = {
    "ridge": "ML model (primary)",
    "lasso": "ML model (alternative)",
    "rf": "ML model (alternative)",
    "xgb": "ML model (alternative)",
    "B1_Momentum10": "Momentum baseline",
    "B3_MomentumDoubleSort10": "Momentum baseline",
    "SPY": "Investable benchmark",
    "QQQ": "Investable benchmark",
}
COLORS = {
    "ridge": "#12355b",
    "lasso": "#d97706",
    "rf": "#2f7d32",
    "xgb": "#b3261e",
    "B1_Momentum10": "#0e7490",
    "B3_MomentumDoubleSort10": "#6d28d9",
    "SPY": "#6b7280",
    "QQQ": "#92400e",
}
MARKERS = {"ridge": "o", "lasso": "s", "rf": "^", "xgb": "D", "B1_Momentum10": "v", "B3_MomentumDoubleSort10": "P", "SPY": None, "QQQ": None}


def _load(cfg: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    main_path = resolve(cfg["paths"]["oos"]) / "oos_monthly_returns.csv"
    ext_path = resolve(cfg["paths"]["ridge_extension"]) / "period_returns.csv"
    if not main_path.exists():
        raise FileNotFoundError(f"Missing OOS monthly returns: {main_path}. Run the OOS stage first.")
    if not ext_path.exists():
        raise FileNotFoundError(f"Missing extension period returns: {ext_path}. Run the extension stage first.")
    main = pd.read_csv(main_path, parse_dates=["decision_date"])
    ext = pd.read_csv(ext_path, parse_dates=["decision_date"])
    return main, ext


def _align_and_check(cfg: Mapping[str, Any], main: pd.DataFrame, ext: pd.DataFrame) -> Dict[str, Any]:
    checks: Dict[str, Any] = {"generated_utc": utc_now()}
    main_dates = pd.DatetimeIndex(sorted(main["decision_date"].unique()))
    ext_dates = pd.DatetimeIndex(sorted(ext["decision_date"].unique()))
    checks["main_decision_dates"] = len(main_dates)
    checks["extension_decision_dates"] = len(ext_dates)
    checks["decision_dates_identical"] = bool(main_dates.equals(ext_dates))
    if not checks["decision_dates_identical"]:
        raise AssertionError("Main and extension backtests do not share the same decision dates")

    benchmark_diffs = {}
    for benchmark in BENCHMARKS:
        a = main.loc[main.strategy == benchmark].set_index("decision_date").net_return.sort_index()
        b = ext.loc[ext.strategy == benchmark].set_index("decision_date").net_return.sort_index()
        joined = pd.concat([a.rename("main"), b.rename("extension")], axis=1, join="inner")
        benchmark_diffs[benchmark] = float((joined["main"] - joined["extension"]).abs().max())
    checks["benchmark_max_abs_difference"] = benchmark_diffs
    checks["benchmarks_identical"] = all(value <= 1e-12 for value in benchmark_diffs.values())
    if not checks["benchmarks_identical"]:
        raise AssertionError(f"Benchmark return conventions differ across engines: {benchmark_diffs}")

    # B3 must be the momentum Top-10 within the same 30-candidate pool that the
    # main model uses.  Verify against the main OOS predictions, not just names.
    predictions_path = resolve(cfg["paths"]["oos"]) / "oos_predictions.csv"
    holdings_path = resolve(cfg["paths"]["ridge_extension"]) / "holdings.csv"
    checks["b3_rule_verified"] = None
    if predictions_path.exists() and holdings_path.exists():
        predictions = pd.read_csv(predictions_path, parse_dates=["decision_date"])
        ridge = predictions.loc[predictions.model == "ridge"]
        holdings = pd.read_csv(holdings_path, parse_dates=["decision_date"])
        b3 = holdings.loc[holdings.strategy == "B3_MomentumDoubleSort10"]
        mismatches = []
        for date, block in ridge.groupby("decision_date"):
            expected = set(block.sort_values(["mom_2_12", "ticker"], ascending=[False, True]).head(10).ticker)
            observed = set(b3.loc[b3.decision_date == date, "ticker"])
            if expected != observed:
                mismatches.append(str(date.date()))
        checks["b3_mismatched_dates"] = mismatches
        checks["b3_rule_verified"] = len(mismatches) == 0
        if mismatches:
            raise AssertionError(f"B3 selection is not momentum Top-10 on the main candidate pool: {mismatches[:3]}")
        checks["b3_candidate_pool"] = "main 3x3 screened 30-stock pool (verified date by date)"
    return checks


def build_panel(cfg: Optional[Mapping[str, Any]] = None) -> tuple[pd.DataFrame, Dict[str, Any]]:
    cfg = load_config() if cfg is None else cfg
    main, ext = _load(cfg)
    checks = _align_and_check(cfg, main, ext)
    frames = [main.loc[main.strategy.isin(ML_MODELS + BENCHMARKS)].copy()]
    frames.append(ext.loc[ext.strategy.isin(MOMENTUM)].copy())
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.loc[panel.strategy.isin(STRATEGIES)]
    panel["decision_date"] = pd.to_datetime(panel.decision_date)
    panel["role"] = panel["strategy"].map(ROLE)
    panel["display"] = panel["strategy"].map(DISPLAY)
    panel = panel.sort_values(["strategy", "decision_date"]).reset_index(drop=True)
    counts = panel.groupby("strategy").size()
    if counts.nunique() != 1:
        raise AssertionError(f"Strategies have different period counts: {counts.to_dict()}")
    checks["periods_per_strategy"] = int(counts.iloc[0])
    checks["n_strategies"] = int(panel.strategy.nunique())
    checks["strategies"] = STRATEGIES
    return panel, checks


def _annual_returns(panel: pd.DataFrame) -> pd.DataFrame:
    wide = panel.pivot(index="decision_date", columns="strategy", values="net_return").sort_index()
    return (1.0 + wide).groupby(wide.index.year).prod() - 1.0


def _metrics(panel: pd.DataFrame, frequency: str) -> pd.DataFrame:
    rows = []
    for strategy in STRATEGIES:
        row = portfolio_metrics(panel, strategy, frequency=frequency, benchmark="SPY")
        row["display"] = DISPLAY[strategy]
        row["role"] = ROLE[strategy]
        row["is_primary"] = strategy == "ridge"
        rows.append(row)
    frame = pd.DataFrame(rows)
    # Reorder columns for readability without dropping any diagnostic.
    front = [
        "strategy",
        "display",
        "role",
        "is_primary",
        "n_periods",
        "total_return_gross",
        "total_return_net",
        "annualized_return",
        "annualized_return_gross",
        "average_period_return",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "calmar_ratio",
        "average_turnover",
        "total_transaction_cost",
        "average_period_transaction_cost",
        "positive_period_share",
        "best_period_return",
        "worst_period_return",
        "benchmark_relative_annualized_return",
        "tracking_error",
        "information_ratio",
        "beta_to_benchmark",
        "alpha_annualized",
    ]
    return frame[front + [c for c in frame.columns if c not in front]]


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_figures(panel: pd.DataFrame, metrics: pd.DataFrame, annual: pd.DataFrame, figures_dir: Path) -> list[Dict[str, str]]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    index: list[Dict[str, str]] = []
    wide = panel.pivot(index="decision_date", columns="strategy", values="net_return").sort_index()

    # Figure 1 - four ML models under the same trading rule
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for strategy in ML_MODELS:
        ax.plot(
            wide.index,
            (1.0 + wide[strategy]).cumprod(),
            label=DISPLAY[strategy],
            color=COLORS[strategy],
            linewidth=2.0 if strategy == "ridge" else 1.5,
        )
    ax.set_title("Out-of-sample growth of $1: four ML models")
    ax.set_ylabel("Growth of $1 (net of costs)")
    ax.set_xlabel("Decision date")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    _save(fig, figures_dir / "fig1_four_model_wealth.png")
    index.append({"figure": "Figure 1", "file": "fig1_four_model_wealth.png", "source": "eight_strategy/monthly_returns.csv"})

    # Figure 2 - primary Ridge vs traditional baselines and benchmarks
    subset = ["ridge", "B1_Momentum10", "B3_MomentumDoubleSort10", "SPY", "QQQ"]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for strategy in subset:
        ax.plot(
            wide.index,
            (1.0 + wide[strategy]).cumprod(),
            label=DISPLAY[strategy],
            color=COLORS[strategy],
            linewidth=2.2 if strategy == "ridge" else 1.4,
            linestyle="-" if strategy not in BENCHMARKS else "--",
        )
    ax.set_title("Primary Ridge strategy vs momentum baselines and investable benchmarks")
    ax.set_ylabel("Growth of $1 (net of costs)")
    ax.set_xlabel("Decision date")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, ncol=2)
    _save(fig, figures_dir / "fig2_ridge_vs_benchmarks_wealth.png")
    index.append({"figure": "Figure 2", "file": "fig2_ridge_vs_benchmarks_wealth.png", "source": "eight_strategy/monthly_returns.csv"})

    # Figure 3 - eight-strategy summary (CAGR and Sharpe)
    order = metrics.sort_values("annualized_return", ascending=True).strategy.tolist()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.4), sharey=True)
    y = np.arange(len(order))
    colors = [COLORS[s] for s in order]
    values = metrics.set_index("strategy").loc[order, "annualized_return"].to_numpy()
    ax1.barh(y, values, color=colors, height=0.65)
    ax1.set_yticks(y)
    ax1.set_yticklabels([DISPLAY[s] for s in order])
    ax1.axvline(0, color="black", linewidth=0.8)
    ax1.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax1.set_title("(a) Annualised return (CAGR, net)")
    ax1.set_xlim(0, float(np.nanmax(values)) * 1.16)
    ax1.grid(axis="x", alpha=0.3)
    for yi, value in zip(y, values):
        ax1.text(value + 0.006 if value >= 0 else value - 0.006, yi, f"{value:.1%}", va="center", ha="left" if value >= 0 else "right", fontsize=8)
    sharpes = metrics.set_index("strategy").loc[order, "sharpe_ratio"].to_numpy()
    ax2.barh(y, sharpes, color=colors, height=0.65)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_title("(b) Sharpe ratio (risk-free = 0)")
    ax2.set_xlim(0, float(np.nanmax(sharpes)) * 1.10)
    ax2.grid(axis="x", alpha=0.3)
    for yi, value in zip(y, sharpes):
        ax2.text(value + 0.02, yi, f"{value:.2f}", va="center", fontsize=8)
    fig.suptitle("Eight-strategy out-of-sample performance summary", fontsize=13)
    fig.tight_layout()
    _save(fig, figures_dir / "fig3_eight_strategy_summary.png")
    index.append({"figure": "Figure 3", "file": "fig3_eight_strategy_summary.png", "source": "eight_strategy/metrics.csv"})

    # Figure 4 - drawdowns
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    for strategy in subset:
        wealth = (1.0 + wide[strategy]).cumprod()
        drawdown = wealth / wealth.cummax() - 1.0
        ax.plot(wide.index, drawdown, label=DISPLAY[strategy], color=COLORS[strategy], linewidth=1.6)
    ax.set_title("Out-of-sample drawdowns: Ridge, momentum baselines and benchmarks")
    ax.set_ylabel("Drawdown from peak")
    ax.set_xlabel("Decision date")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, ncol=2)
    _save(fig, figures_dir / "fig4_oos_drawdowns.png")
    index.append({"figure": "Figure 4", "file": "fig4_oos_drawdowns.png", "source": "eight_strategy/monthly_returns.csv"})

    # Figure 5 - annual returns
    years = annual.index.tolist()
    x = np.arange(len(years))
    width = 0.16
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    for position, strategy in enumerate(subset):
        offsets = x + (position - 2) * width
        ax.bar(offsets, annual[strategy].to_numpy(), width, label=DISPLAY[strategy], color=COLORS[strategy])
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_title("Annual net returns: Ridge, momentum baselines and benchmarks")
    ax.set_ylabel("Net return")
    ax.set_xlabel("Year")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, ncol=5, fontsize=8)
    _save(fig, figures_dir / "fig5_annual_returns.png")
    index.append({"figure": "Figure 5", "file": "fig5_annual_returns.png", "source": "eight_strategy/annual_returns.csv"})

    # Figure 6 - risk-adjusted performance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0))
    ordered = metrics.sort_values("sharpe_ratio", ascending=False)
    ax1.barh(ordered.display, ordered.sharpe_ratio, color=[COLORS[s] for s in ordered.strategy])
    ax1.axvline(0, color="black", linewidth=0.8)
    ax1.set_title("(a) Sharpe ratio")
    ax1.grid(axis="x", alpha=0.3)
    for yi, value in enumerate(ordered.sharpe_ratio):
        ax1.text(value + 0.02, yi, f"{value:.2f}", va="center", fontsize=8)
    ordered_vol = metrics.sort_values("annualized_volatility", ascending=False)
    ax2.barh(ordered_vol.display, ordered_vol.annualized_volatility, color=[COLORS[s] for s in ordered_vol.strategy])
    ax2.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax2.set_title("(b) Annualised volatility")
    ax2.grid(axis="x", alpha=0.3)
    for yi, value in enumerate(ordered_vol.annualized_volatility):
        ax2.text(value + 0.004, yi, f"{value:.1%}", va="center", fontsize=8)
    fig.suptitle("Risk-adjusted performance comparison", fontsize=13)
    fig.tight_layout()
    _save(fig, figures_dir / "fig6_risk_adjusted.png")
    index.append({"figure": "Figure 6", "file": "fig6_risk_adjusted.png", "source": "eight_strategy/metrics.csv"})
    return index


def run_eight_strategy_comparison(
    cfg: Optional[Mapping[str, Any]] = None,
    frequency: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = load_config() if cfg is None else cfg
    frequency = frequency or cfg["project"]["frequency"]
    ensure_output_dirs(cfg)
    out = resolve(cfg["paths"]["eight_strategy"])
    out.mkdir(parents=True, exist_ok=True)
    figures_dir = resolve(cfg["paths"]["oos_figures"])

    panel, checks = build_panel(cfg)
    metrics = _metrics(panel, frequency)
    annual = _annual_returns(panel)

    panel.to_csv(out / "monthly_returns.csv", index=False)
    metrics.to_csv(out / "metrics.csv", index=False)
    annual.to_csv(out / "annual_returns.csv")
    comparison_columns = [
        "strategy",
        "display",
        "role",
        "is_primary",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "calmar_ratio",
        "total_return_net",
        "average_turnover",
        "total_transaction_cost",
        "benchmark_relative_annualized_return",
        "information_ratio",
    ]
    metrics[comparison_columns].to_csv(out / "comparison.csv", index=False)
    figures = make_figures(panel, metrics, annual, figures_dir)
    pd.DataFrame(figures).to_csv(out / "figure_index.csv", index=False)

    write_json(
        out / "alignment_checks.json",
        {
            **checks,
            "return_conventions": {
                "decision_dates": "last trading session of each month",
                "execution": "next session open",
                "label_horizon": "execution open to next rebalance execution open",
                "cost_bps_per_side": int(cfg["portfolio"]["cost_bps_per_side"]),
                "benchmark_cost": "one-time entry fee only",
                "annualization": "12 monthly periods",
                "risk_free_rate": 0.0,
                "returns": "net of transaction costs",
            },
            "note": (
                "B1/B3 come from the preserved extension driver; the four ML models and the "
                "benchmarks come from the main backtest. Alignment, benchmark equality and the "
                "B3 selection rule are verified before the panel is built."
            ),
        },
    )
    write_json(
        out / "comparison_metadata.json",
        {
            "generated_utc": utc_now(),
            "n_strategies": len(STRATEGIES),
            "n_periods": int(panel.decision_date.nunique()),
            "date_start": str(panel.decision_date.min().date()),
            "date_end": str(panel.decision_date.max().date()),
            "primary_strategy": "ridge",
            "frequency": frequency,
        },
    )
    return {"panel": panel, "metrics": metrics, "annual": annual, "checks": checks, "figures": figures}
