"""Professional charts generated exclusively from saved result tables.

No performance number is hard-coded: every figure reads the CSVs produced by
:mod:`src.validation`, :mod:`src.model_selection` and :mod:`src.oos_evaluation`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from .data_pipeline import load_config, resolve
from .models import MODEL_LABELS, MODEL_NAMES

MODEL_COLORS = {
    "ridge": "#1f77b4",
    "lasso": "#ff7f0e",
    "rf": "#2ca02c",
    "xgb": "#d62728",
    "SPY": "#7f7f7f",
    "QQQ": "#9467bd",
}

PREDICTION_METRICS = [
    "mse",
    "rmse",
    "mae",
    "r2_sklearn",
    "r2_oos_trainmean",
    "directional_accuracy",
    "directional_accuracy_cs",
    "pearson_corr",
    "mean_rank_ic",
]

TRADING_METRICS = [
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "average_turnover",
    "total_transaction_cost",
]


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _annotate_bars(ax, bars, fmt="{:.4f}") -> None:
    for bar in bars:
        height = bar.get_height()
        if not np.isfinite(height):
            continue
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _zscore_columns(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.astype(float)
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=0).replace(0, np.nan)
    return (values - mean) / std


def _heatmap(
    ax,
    values: pd.DataFrame,
    annot: Optional[pd.DataFrame] = None,
    cmap: str = "RdYlGn",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    fmt: str = "{:.3f}",
):
    data = values.to_numpy(dtype=float)
    image = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(values.shape[1]))
    ax.set_xticklabels(values.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(values.shape[0]))
    ax.set_yticklabels(values.index, fontsize=9)
    if annot is not None:
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                raw = annot.iloc[i, j]
                if pd.isna(raw):
                    continue
                ax.text(j, i, fmt.format(raw), ha="center", va="center", fontsize=7, color="black")
    return image


def plot_validation_dashboard(
    cfg: Optional[Mapping[str, Any]] = None,
    validation_dir: Optional[Path] = None,
    figures_dir: Optional[Path] = None,
) -> list[Path]:
    cfg = load_config() if cfg is None else cfg
    validation_dir = Path(validation_dir) if validation_dir else resolve(cfg["paths"]["validation"])
    figures_dir = Path(figures_dir) if figures_dir else resolve(cfg["paths"]["validation_figures"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    comparison = pd.read_csv(validation_dir / "validation_model_comparison.csv")
    prediction = pd.read_csv(validation_dir / "validation_prediction_metrics.csv")
    portfolio = pd.read_csv(validation_dir / "validation_portfolio_metrics.csv")
    monthly = pd.read_csv(validation_dir / "validation_monthly_returns.csv", parse_dates=["decision_date"])

    comparison = comparison.set_index("model")
    model_order = [m for m in MODEL_NAMES if m in comparison.index]
    bench_order = [b for b in ("SPY", "QQQ") if b in set(portfolio.strategy)]
    portfolio = portfolio.set_index("strategy")
    labels = [MODEL_LABELS[m] for m in model_order]
    produced: list[Path] = []

    def add(name: str) -> Path:
        path = figures_dir / name
        produced.append(path)
        return path

    # 1. Validation MSE bar chart -------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.2))
    values = comparison.loc[model_order, "validation_mse"]
    bars = ax.bar(labels, values, color=[MODEL_COLORS[m] for m in model_order])
    _annotate_bars(ax, bars, "{:.5f}")
    winner = values.idxmin()
    ax.set_title(f"Validation MSE (2020-2021) - lowest: {MODEL_LABELS[winner]}")
    ax.set_ylabel("MSE (lower is better)")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, add("01_validation_mse_bar.png"))

    # 2. RMSE and MAE comparison --------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = np.arange(len(model_order))
    width = 0.36
    rmse = comparison.loc[model_order, "validation_rmse"]
    mae = comparison.loc[model_order, "validation_mae"]
    b1 = ax.bar(x - width / 2, rmse, width, label="RMSE", color="#4c72b0")
    b2 = ax.bar(x + width / 2, mae, width, label="MAE", color="#dd8452")
    _annotate_bars(ax, b1, "{:.4f}")
    _annotate_bars(ax, b2, "{:.4f}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_title("Validation RMSE and MAE by model")
    ax.set_ylabel("Error (decimal return units)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, add("02_rmse_mae_comparison.png"))

    # 3. Validation Sharpe ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.2))
    names = model_order + bench_order
    sharpes = portfolio.loc[names, "sharpe_ratio"]
    bars = ax.bar([MODEL_LABELS.get(n, n) for n in names], sharpes, color=[MODEL_COLORS[n] for n in names])
    _annotate_bars(ax, bars, "{:.3f}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Validation Sharpe ratio (zero risk-free approximation, post-cost)")
    ax.set_ylabel("Annualised Sharpe")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, add("03_validation_sharpe_bar.png"))

    # 4. Annualized return ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.2))
    values = portfolio.loc[names, "annualized_return"]
    bars = ax.bar([MODEL_LABELS.get(n, n) for n in names], values, color=[MODEL_COLORS[n] for n in names])
    _annotate_bars(ax, bars, "{:.2%}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Validation CAGR (post-cost)")
    ax.set_ylabel("Annualised return")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(axis="y", alpha=0.3)
    _save(fig, add("04_annualized_return_bar.png"))

    # 5. Annualized volatility ----------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.2))
    values = portfolio.loc[names, "annualized_volatility"]
    bars = ax.bar([MODEL_LABELS.get(n, n) for n in names], values, color=[MODEL_COLORS[n] for n in names])
    _annotate_bars(ax, bars, "{:.2%}")
    ax.set_title("Validation annualised volatility")
    ax.set_ylabel("Annualised volatility")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(axis="y", alpha=0.3)
    _save(fig, add("05_annualized_volatility_bar.png"))

    # 6. Risk-return scatter -------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name in names:
        row = portfolio.loc[name]
        ax.scatter(
            row["annualized_volatility"],
            row["annualized_return"],
            s=90,
            color=MODEL_COLORS[name],
            label=MODEL_LABELS.get(name, name),
            edgecolor="black",
            linewidth=0.5,
        )
        ax.annotate(MODEL_LABELS.get(name, name), (row["annualized_volatility"], row["annualized_return"]), xytext=(6, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Annualised volatility")
    ax.set_ylabel("Annualised return (CAGR)")
    ax.set_title("Validation risk-return map")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(alpha=0.3)
    _save(fig, add("06_risk_return_scatter.png"))

    # 7. Cumulative wealth ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name in names:
        series = monthly.loc[monthly.strategy == name].sort_values("decision_date")
        ax.plot(series.decision_date, (1.0 + series.net_return).cumprod(), label=MODEL_LABELS.get(name, name), color=MODEL_COLORS[name], linewidth=1.8 if name in model_order else 1.2, linestyle="-" if name in model_order else "--")
    ax.set_title("Validation growth of $1 (net of 20 bps per traded side)")
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("Decision date")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, add("07_cumulative_wealth.png"))

    # 8. Drawdown comparison -------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5))
    for name in names:
        series = monthly.loc[monthly.strategy == name].sort_values("decision_date")
        wealth = (1.0 + series.net_return).cumprod()
        drawdown = wealth / wealth.cummax() - 1.0
        ax.plot(series.decision_date, drawdown, label=MODEL_LABELS.get(name, name), color=MODEL_COLORS[name], linewidth=1.4)
    ax.set_title("Validation drawdown paths")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, add("08_drawdown_comparison.png"))

    # 9. Monthly returns heatmap --------------------------------------------
    wide = monthly.pivot(index="strategy", columns="decision_date", values="net_return")
    wide = wide.reindex([n for n in names if n in wide.index])
    fig, ax = plt.subplots(figsize=(14, 4.6))
    image = ax.imshow(wide.to_numpy(dtype=float), aspect="auto", cmap="RdYlGn", vmin=-0.2, vmax=0.2)
    ax.set_yticks(range(len(wide.index)))
    ax.set_yticklabels([MODEL_LABELS.get(n, n) for n in wide.index])
    ax.set_xticks(range(wide.shape[1]))
    ax.set_xticklabels([pd.Timestamp(c).strftime("%Y-%m") for c in wide.columns], rotation=90, fontsize=7)
    ax.set_title("Validation monthly net returns")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.01, label="Net return")
    _save(fig, add("09_monthly_returns_heatmap.png"))

    # 10. Prediction metrics heatmap ----------------------------------------
    pred = prediction.set_index("model").reindex(model_order)
    available = [c for c in PREDICTION_METRICS if c in pred.columns]
    raw = pred[available].astype(float)
    z = _zscore_columns(raw).fillna(0.0)
    fig, ax = plt.subplots(figsize=(12, 4.2))
    image = _heatmap(ax, z, annot=raw, cmap="RdYlGn")
    ax.set_title("Validation prediction metrics (colour = column z-score, text = raw value)")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.01, label="Column z-score")
    _save(fig, add("10_prediction_metrics_heatmap.png"))

    # 11. Trading metrics heatmap -------------------------------------------
    trading = portfolio.reindex(names)
    available = [c for c in TRADING_METRICS if c in trading.columns]
    raw = trading[available].astype(float)
    z = _zscore_columns(raw).fillna(0.0)
    fig, ax = plt.subplots(figsize=(13, 5))
    image = _heatmap(ax, z, annot=raw, cmap="RdYlGn")
    ax.set_title("Validation trading metrics (colour = column z-score, text = raw value)")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.01, label="Column z-score")
    _save(fig, add("11_trading_metrics_heatmap.png"))

    # 12. MSE vs Sharpe scatter ---------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name in model_order:
        ax.scatter(comparison.loc[name, "validation_mse"], comparison.loc[name, "validation_sharpe"], s=110, color=MODEL_COLORS[name], edgecolor="black", linewidth=0.5)
        ax.annotate(MODEL_LABELS[name], (comparison.loc[name, "validation_mse"], comparison.loc[name, "validation_sharpe"]), xytext=(6, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Validation MSE (lower = more accurate)")
    ax.set_ylabel("Validation Sharpe (higher = better portfolio)")
    ax.set_title("Predictive accuracy vs investment performance")
    ax.grid(alpha=0.3)
    _save(fig, add("12_mse_vs_sharpe_scatter.png"))

    # 13. Turnover and transaction costs ------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    turnover = portfolio.loc[model_order, "average_turnover"]
    cost = portfolio.loc[model_order, "total_transaction_cost"]
    b1 = ax1.bar(labels, turnover, color=[MODEL_COLORS[m] for m in model_order])
    _annotate_bars(ax1, b1, "{:.3f}")
    ax1.set_title("Average monthly turnover")
    ax1.set_ylabel("sum |weight change|")
    ax1.grid(axis="y", alpha=0.3)
    b2 = ax2.bar(labels, cost, color=[MODEL_COLORS[m] for m in model_order])
    _annotate_bars(ax2, b2, "{:.4f}")
    ax2.set_title("Total transaction-cost drag over validation")
    ax2.set_ylabel("Cumulative cost (return units)")
    ax2.grid(axis="y", alpha=0.3)
    for axis in (ax1, ax2):
        axis.set_xticklabels(labels, rotation=15)
    _save(fig, add("13_turnover_cost_comparison.png"))

    # 14. Model ranking summary ---------------------------------------------
    rank_frame = comparison.loc[model_order, ["mse_rank", "sharpe_rank", "cagr_rank", "max_drawdown_rank"]].astype(float)
    rank_frame.columns = ["Lowest MSE", "Highest Sharpe", "Highest CAGR", "Lowest Max DD"]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    image = _heatmap(ax, rank_frame, annot=rank_frame, cmap="RdYlGn_r", vmin=1, vmax=len(model_order), fmt="{:.0f}")
    ax.set_yticklabels(labels)
    ax.set_title("Model rankings (1 = best; separate criteria are not interchangeable)")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.01, label="Rank")
    _save(fig, add("14_model_ranking_summary.png"))

    # 15. Benchmark comparison ----------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    values = portfolio.loc[names, "annualized_return"]
    bars = ax1.bar([MODEL_LABELS.get(n, n) for n in names], values, color=[MODEL_COLORS[n] for n in names])
    _annotate_bars(ax1, bars, "{:.2%}")
    ax1.axhline(0, color="black", linewidth=0.8)
    for benchmark in bench_order:
        ax1.axhline(portfolio.loc[benchmark, "annualized_return"], color=MODEL_COLORS[benchmark], linestyle="--", linewidth=1.0, label=f"{benchmark} buy-and-hold")
    ax1.set_title("Validation CAGR vs buy-and-hold benchmarks")
    ax1.set_ylabel("Annualised return")
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax1.legend(fontsize=7)
    ax1.grid(axis="y", alpha=0.3)
    values = portfolio.loc[names, "sharpe_ratio"]
    bars = ax2.bar([MODEL_LABELS.get(n, n) for n in names], values, color=[MODEL_COLORS[n] for n in names])
    _annotate_bars(ax2, bars, "{:.2f}")
    ax2.axhline(0, color="black", linewidth=0.8)
    for benchmark in bench_order:
        ax2.axhline(portfolio.loc[benchmark, "sharpe_ratio"], color=MODEL_COLORS[benchmark], linestyle="--", linewidth=1.0, label=f"{benchmark} buy-and-hold")
    ax2.set_title("Validation Sharpe vs benchmarks")
    ax2.set_ylabel("Sharpe (rf=0)")
    ax2.legend(fontsize=7)
    ax2.grid(axis="y", alpha=0.3)
    for axis in (ax1, ax2):
        axis.set_xticklabels([MODEL_LABELS.get(n, n) for n in names], rotation=20, fontsize=8)
    _save(fig, add("15_benchmark_comparison.png"))

    return produced


def plot_oos_dashboard(
    cfg: Optional[Mapping[str, Any]] = None,
    oos_dir: Optional[Path] = None,
    figures_dir: Optional[Path] = None,
) -> list[Path]:
    cfg = load_config() if cfg is None else cfg
    oos_dir = Path(oos_dir) if oos_dir else resolve(cfg["paths"]["oos"])
    figures_dir = Path(figures_dir) if figures_dir else resolve(cfg["paths"]["oos_figures"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    monthly = pd.read_csv(oos_dir / "oos_monthly_returns.csv", parse_dates=["decision_date"])
    portfolio = pd.read_csv(oos_dir / "oos_portfolio_metrics.csv").set_index("strategy")
    comparison = pd.read_csv(oos_dir / "oos_model_comparison.csv").set_index("model")
    degradation = pd.read_csv(oos_dir / "validation_to_oos_degradation.csv")

    model_order = [m for m in MODEL_NAMES if m in set(monthly.strategy)]
    bench_order = [b for b in ("SPY", "QQQ") if b in set(monthly.strategy)]
    names = model_order + bench_order
    selected = comparison.loc[comparison["is_selected"].astype(bool), "is_selected"].index.tolist() if "is_selected" in comparison else []
    produced: list[Path] = []

    def add(name: str) -> Path:
        path = figures_dir / name
        produced.append(path)
        return path

    fig, ax = plt.subplots(figsize=(12, 6))
    for name in names:
        series = monthly.loc[monthly.strategy == name].sort_values("decision_date")
        ax.plot(
            series.decision_date,
            (1.0 + series.net_return).cumprod(),
            label=MODEL_LABELS.get(name, name) + (" (selected)" if name in selected else ""),
            color=MODEL_COLORS[name],
            linewidth=2.0 if name in model_order else 1.2,
            linestyle="-" if name in model_order else "--",
        )
    ax.set_title("Out-of-sample growth of $1, 2022-2025 (net of costs)")
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("Decision date")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, add("01_oos_cumulative_wealth.png"))

    fig, ax = plt.subplots(figsize=(12, 5))
    for name in names:
        series = monthly.loc[monthly.strategy == name].sort_values("decision_date")
        wealth = (1.0 + series.net_return).cumprod()
        ax.plot(series.decision_date, wealth / wealth.cummax() - 1.0, label=MODEL_LABELS.get(name, name), color=MODEL_COLORS[name], linewidth=1.4)
    ax.set_title("Out-of-sample drawdowns, 2022-2025")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, add("02_oos_drawdowns.png"))

    wide = monthly.pivot(index="strategy", columns="decision_date", values="net_return").reindex([n for n in names if n in set(monthly.strategy)])
    years = pd.Index(wide.columns).year
    annual = (1.0 + wide).T.groupby(years).prod().T - 1.0
    fig, ax = plt.subplots(figsize=(11, 4.8))
    annual.T.plot(kind="bar", ax=ax, width=0.82)
    ax.set_title("Out-of-sample annual net returns by model")
    ax.set_ylabel("Net return")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(ncol=3, fontsize=7)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    fig.tight_layout()
    fig.savefig(figures_dir / "03_oos_annual_returns.png", dpi=170)
    plt.close(fig)
    produced.append(figures_dir / "03_oos_annual_returns.png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    values = portfolio.loc[names, "annualized_return"]
    bars = ax1.bar([MODEL_LABELS.get(n, n) for n in names], values, color=[MODEL_COLORS[n] for n in names])
    _annotate_bars(ax1, bars, "{:.2%}")
    ax1.axhline(0, color="black", linewidth=0.8)
    for benchmark in bench_order:
        ax1.axhline(portfolio.loc[benchmark, "annualized_return"], color=MODEL_COLORS[benchmark], linestyle="--", linewidth=1.0, label=f"{benchmark} buy-and-hold")
    ax1.set_title("OOS CAGR vs benchmarks")
    ax1.set_ylabel("Annualised return")
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax1.legend(fontsize=7)
    ax1.grid(axis="y", alpha=0.3)
    values = portfolio.loc[names, "sharpe_ratio"]
    bars = ax2.bar([MODEL_LABELS.get(n, n) for n in names], values, color=[MODEL_COLORS[n] for n in names])
    _annotate_bars(ax2, bars, "{:.2f}")
    ax2.axhline(0, color="black", linewidth=0.8)
    for benchmark in bench_order:
        ax2.axhline(portfolio.loc[benchmark, "sharpe_ratio"], color=MODEL_COLORS[benchmark], linestyle="--", linewidth=1.0, label=f"{benchmark} buy-and-hold")
    ax2.set_title("OOS Sharpe vs benchmarks")
    ax2.set_ylabel("Sharpe (rf=0)")
    ax2.legend(fontsize=7)
    ax2.grid(axis="y", alpha=0.3)
    for axis in (ax1, ax2):
        axis.set_xticklabels([MODEL_LABELS.get(n, n) for n in names], rotation=20, fontsize=8)
    _save(fig, add("04_oos_benchmark_comparison.png"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    order = degradation["model"].tolist()
    x = np.arange(len(order))
    ax1.bar(x - 0.2, degradation["validation_mse"], 0.4, label="Validation", color="#4c72b0")
    ax1.bar(x + 0.2, degradation["oos_mse"], 0.4, label="OOS", color="#dd8452")
    ax1.set_xticks(x)
    ax1.set_xticklabels([MODEL_LABELS.get(m, m) for m in order], rotation=15, fontsize=8)
    ax1.set_title("MSE: validation vs OOS")
    ax1.set_ylabel("MSE")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    ax2.bar(x - 0.2, degradation["validation_sharpe"], 0.4, label="Validation", color="#4c72b0")
    ax2.bar(x + 0.2, degradation["oos_sharpe"], 0.4, label="OOS", color="#dd8452")
    ax2.set_xticks(x)
    ax2.set_xticklabels([MODEL_LABELS.get(m, m) for m in order], rotation=15, fontsize=8)
    ax2.set_title("Sharpe: validation vs OOS")
    ax2.set_ylabel("Sharpe (rf=0)")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    _save(fig, add("05_validation_to_oos_degradation.png"))

    return produced
