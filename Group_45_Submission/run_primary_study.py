"""Compare four models on validation, then refit and test Ridge with benchmarks."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest import portfolio_backtest
from src.data_pipeline import (
    build_extension_panel,
    chronological_splits,
    load_config,
    load_extension_panel,
    load_model_ready,
    pipeline_config,
    rebuild_model_ready,
    resolve,
)
from src.metrics import portfolio_metrics, prediction_metrics
from src.models import build_model
from src.pipeline import FEATURES, download
from src.utils import write_json
from src.validation import run_validation

ROOT = Path(__file__).resolve().parent
LABEL = "target_forward_return"
STRATEGIES = {
    "ridge": "Ridge Top 10",
    "B1": "B1: broad momentum",
    "B3": "B3: same-pool momentum",
    "SPY": "SPY",
    "QQQ": "QQQ",
}


def build_data(cfg):
    """Download the vendor inputs and regenerate both monthly derived panels."""
    download(pipeline_config(cfg))
    rebuild_model_ready(cfg)
    build_extension_panel(cfg, ROOT, force=True)


def record_model_selection(cfg, validation, model, rationale):
    comparison = validation["comparison"].set_index("model")
    parameters = validation["best_configs"][model]["params"]
    selected = validation["search"].loc[lambda rows: rows.selected].set_index("model")
    decision = {
        "model": model,
        "parameters": parameters,
        "within_model_rule": "minimum validation MSE",
        "model_rationale": rationale,
        "validation_rank_ic": float(comparison.at[model, "validation_mean_rank_ic"]),
        "validation_mse_winner": str(comparison.validation_mse.idxmin()),
        "lasso_zero_coefficients": len(FEATURES) - int(selected.at["lasso", "convergence_n_nonzero_features"]),
        "validation_dates": int(validation["metadata"]["validation_dates"]),
    }
    write_json(resolve(cfg["paths"]["validation"]) / "primary_decision.json", decision)
    return decision


def broad_momentum_returns(cfg, decision_dates):
    liquid = load_extension_panel(cfg).loc[lambda rows: rows.decision_date.isin(decision_dates)]
    returns, _, _ = portfolio_backtest(
        liquid.assign(prediction=liquid.mom_2_12),
        cfg["project"]["frequency"],
        cfg["portfolio"]["cost_bps_per_side"],
        cfg["portfolio"]["top_n"],
    )
    return returns.loc[returns.strategy.eq("model_top10")].assign(strategy="B1")


def compare_validation_strategies(cfg, validation):
    candidate_columns = ["decision_date", "ticker", "mom_2_12", LABEL, "spy_return", "qqq_return"]
    scored = validation["predictions"][candidate_columns].drop_duplicates(["decision_date", "ticker"])
    dates = pd.DatetimeIndex(pd.to_datetime(scored.decision_date.unique())).sort_values()
    frequency = cfg["project"]["frequency"]
    returns, _, _ = portfolio_backtest(
        scored.assign(prediction=scored.mom_2_12),
        frequency, cfg["portfolio"]["cost_bps_per_side"], cfg["portfolio"]["top_n"],
    )
    b3 = returns.loc[returns.strategy.eq("momentum_top10")].assign(strategy="B3")
    b1 = broad_momentum_returns(cfg, dates)
    panel = pd.concat([validation["monthly_returns"], b1, b3], ignore_index=True)
    panel["decision_date"] = pd.to_datetime(panel.decision_date)
    order = ["ridge", "lasso", "rf", "xgb", "B1", "B3", "SPY", "QQQ"]
    aligned = panel.pivot(index="decision_date", columns="strategy", values="net_return")[order]
    if not aligned.index.equals(dates) or aligned.isna().any().any():
        raise ValueError("Validation strategies must share the same settled decision dates.")

    forecasts = validation["comparison"].set_index("model")[
        ["validation_mse", "validation_mean_rank_ic"]
    ].rename(columns={"validation_mse": "mse", "validation_mean_rank_ic": "mean_rank_ic"})
    forecasts.loc["B3", "mean_rank_ic"] = scored.groupby("decision_date").apply(
        lambda month: month.mom_2_12.corr(month[LABEL], method="spearman"), include_groups=False,
    ).mean()
    summary = pd.DataFrame([
        portfolio_metrics(panel, name, frequency=frequency, benchmark="SPY") for name in order
    ]).join(forecasts, on="strategy")
    out = resolve(cfg["paths"]["validation"])
    summary.to_csv(out / "validation_strategy_comparison.csv", index=False)
    panel.sort_values(["strategy", "decision_date"]).to_csv(
        out / "validation_strategy_monthly_returns.csv", index=False,
    )
    return summary


def evaluate_ridge(cfg, decision):
    splits, _, _ = chronological_splits(load_model_ready(cfg), cfg)
    refit, test = splits["refit"], splits["test"].copy()
    estimator = build_model(decision["model"], decision["parameters"], cfg["project"]["random_seed"])
    estimator.fit(refit[FEATURES], refit[LABEL])
    test["prediction"] = estimator.predict(test[FEATURES])
    target = test[LABEL]
    naive = float(refit[LABEL].mean())
    forecast = prediction_metrics(target, test.prediction, naive, dates=test.decision_date)
    frequency = cfg["project"]["frequency"]
    cost = cfg["portfolio"]["cost_bps_per_side"]
    top_n = cfg["portfolio"]["top_n"]

    returns, _, _ = portfolio_backtest(test, frequency, cost, top_n)
    returns = returns.loc[returns.strategy.ne("equal_weight_30")].copy()
    returns.strategy = returns.strategy.replace({"model_top10": "ridge", "momentum_top10": "B3"})

    broad = broad_momentum_returns(cfg, test.decision_date)
    panel = pd.concat([returns, broad], ignore_index=True)
    panel["decision_date"] = pd.to_datetime(panel.decision_date)
    wide = panel.pivot(index="decision_date", columns="strategy", values="net_return").sort_index()
    wide = wide[list(STRATEGIES)]
    if wide.isna().any().any():
        raise ValueError("Portfolio and benchmark decision dates do not match.")
    periods = test[["decision_date", "execution_date", "label_end_date"]].drop_duplicates().sort_values("decision_date")
    if not np.array_equal(periods.label_end_date.iloc[:-1].to_numpy(), periods.execution_date.iloc[1:].to_numpy()):
        raise ValueError("Holding periods must meet at the rebalance open.")

    metrics = pd.DataFrame([portfolio_metrics(panel, name, frequency=frequency, benchmark="SPY") for name in STRATEGIES])
    annual = (1 + wide).groupby(wide.index.year).prod() - 1
    annual.index.name = "decision_year"
    annual_sharpe = np.sqrt(12) * wide.groupby(wide.index.year).mean() / wide.groupby(wide.index.year).std()
    annual_sharpe.index.name = "decision_year"

    out = resolve(cfg["paths"]["oos"])
    out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / "portfolio_metrics.csv", index=False)
    panel.to_csv(out / "monthly_returns.csv", index=False)
    annual.to_csv(out / "annual_returns.csv")
    annual_sharpe.to_csv(out / "annual_sharpe.csv")
    test[["decision_date", "execution_date", "label_end_date", "ticker", "prediction", LABEL]].to_csv(
        out / "ridge_predictions.csv", index=False
    )
    write_json(out / "ridge_forecast_metrics.json", {
        **forecast, "refit_rows": len(refit), "first_entry": str(periods.execution_date.min().date()),
        "last_exit": str(periods.label_end_date.max().date()), "historical_mean": naive,
    })
    return metrics, wide, annual, annual_sharpe, forecast


def plot_results(wide, annual, annual_sharpe, forecast, directory):
    directory.mkdir(parents=True, exist_ok=True)
    colors = ["#1f77b4", "#ff7f0e", "#169c8f", "#666666", "#7461a8"]
    styles = ["-", "--", "-.", ":", (0, (5, 2, 1, 2))]
    wealth = (1 + wide).cumprod()
    wealth.loc[wealth.index.min() - pd.offsets.MonthEnd(1)] = 1.0
    wealth = wealth.sort_index()
    fig, ax = plt.subplots(figsize=(8.6, 4.4), layout="constrained")
    for (name, label), color, style in zip(STRATEGIES.items(), colors, styles):
        ax.plot(wealth.index, wealth[name], color=color, linestyle=style, linewidth=1.8, label=label)
    ax.set(xlabel="Decision date", ylabel="Net wealth (initial value = 1)")
    ax.grid(axis="y", alpha=.20)
    ax.legend(frameon=False, ncol=2)
    fig.savefig(directory / "ridge_benchmarks.png", dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8), layout="constrained")
    names = ["ridge", "B1", "B3"]
    for ax, frame, ylabel in [(axes[0], annual * 100, "Net return (%)"),
                              (axes[1], annual_sharpe, "Annualised Sharpe (RF = 0)")]:
        frame[names].rename(columns=STRATEGIES).plot.bar(ax=ax, color=colors[:3], legend=False)
        ax.set(xlabel="Decision year", ylabel=ylabel)
        ax.axhline(0, linewidth=.7, color="#444444")
        ax.tick_params(axis="x", rotation=0)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(directory / "annual_returns_sharpe.png", dpi=190)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 1.9), layout="constrained")
    value = 100 * forecast["r2_oos_trainmean"]
    ax.barh(["Ridge"], [value], color=colors[0], height=.35)
    ax.text(value + .04, 0, f"{value:.2f}%", va="center")
    ax.axvline(0, linewidth=.7, color="#444444")
    ax.set(xlim=(-.15, 2.45), xlabel="Out-of-sample R² (%) relative to the pre-test mean")
    fig.savefig(directory / "ridge_oos_r2.png", dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild-data", action="store_true", help="Refresh Yahoo data and rebuild the monthly panels.")
    args = parser.parse_args()
    cfg = load_config()
    if args.rebuild_data:
        build_data(cfg)
    validation = run_validation(cfg, criterion="lowest_validation_mse")
    comparison = compare_validation_strategies(cfg, validation)
    choice = record_model_selection(
        cfg, validation, "ridge",
        "Highest validation Rank IC and joint L2 shrinkage of correlated predictors.",
    )
    metrics, wide, annual, annual_sharpe, forecast = evaluate_ridge(cfg, choice)
    plot_results(wide, annual, annual_sharpe, forecast, ROOT / "figures")
    print(comparison[["strategy", "mse", "mean_rank_ic", "annualized_return", "sharpe_ratio"]].to_string(index=False))
    print(metrics[["strategy", "annualized_return", "sharpe_ratio", "max_drawdown"]].to_string(index=False))
