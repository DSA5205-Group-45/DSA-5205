"""Nine-strategy Ridge extension: B1-B3, R1-R4, SPY, QQQ.

This module is the executable counterpart of the authoritative notebook
``03_ridge_nine_strategy_comparison.ipynb``.  It imports the original
``strategies.py`` (copied verbatim) and calls ``pools``, ``train_ridge``,
``targets_for_date``, ``simulate`` and ``performance`` unchanged.  No signal,
weight, execution or cost rule is re-implemented here.

The extension is intentionally a *separate* analysis from the four-model
comparison:

* four-model comparison  -> same trading strategy, four forecasting algorithms;
* nine-strategy extension -> Ridge-related strategy variants plus benchmarks.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data_pipeline import ensure_output_dirs, load_config, load_extension_panel, resolve
from .pipeline import FEATURES
from .utils import environment_info, utc_now, write_json

# Import the preserved extension module from the project root.
import strategies as ext

NINE_STRATEGIES = [
    "B1_Momentum10",
    "B2_Momentum30",
    "B3_MomentumDoubleSort10",
    "R1_RidgeWeighted10",
    "R2_RidgeFree60",
    "R3_RidgeQuota343",
    "R4_RidgeDoubleSort30",
    "SPY",
    "QQQ",
]
# The former R4_RidgeDoubleSort30 is the same economic rule as Main Ridge. It is
# retained internally for reproducibility but is not presented as a separate
# strategy in the final research materials.
PUBLIC_STRATEGIES = [s for s in NINE_STRATEGIES if s != "R4_RidgeDoubleSort30"]
RIDGE_STRATEGIES = [s for s in NINE_STRATEGIES if s.startswith("R")]
MOMENTUM_STRATEGIES = [s for s in NINE_STRATEGIES if s.startswith("B")]
BENCHMARKS = ["SPY", "QQQ"]


def assert_nine_strategies(selections: Mapping[Any, Mapping[str, Any]]) -> None:
    for date, groups in selections.items():
        if set(groups) != set(MOMENTUM_STRATEGIES + RIDGE_STRATEGIES):
            raise AssertionError(f"Unexpected strategy set on {date}: {sorted(groups)}")


def _build_pools(panel: pd.DataFrame, frequency: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rebuild the ADV20 Top-300, s60 and s30 candidate pools period by period."""
    sixties, thirties = [], []
    for date, g in panel.groupby("decision_date", sort=True):
        s60, s30 = ext.pools(g)
        s60 = s60.assign(decision_date=date)
        s30 = s30.assign(decision_date=date)
        sixties.append(s60)
        thirties.append(s30)
    s60 = pd.concat(sixties, ignore_index=True)
    s30 = pd.concat(thirties, ignore_index=True)
    # ``pools`` is a pure selection function over the candidate rows, so the
    # label columns travel with them.  Only the decision-date alias is added.
    for frame in (s60, s30):
        frame["decision_date"] = pd.to_datetime(frame["date"])
        frame["label_end_date"] = pd.to_datetime(frame["label_end_date"])
        missing = {"target_forward_return", "execution_date", "spy_return", "qqq_return"} - set(frame.columns)
        if missing:
            raise ValueError(f"Extension pool is missing label columns: {sorted(missing)}")
    return panel, s60, s30


def run_ridge_extension(
    cfg: Optional[Mapping[str, Any]] = None,
    frequency: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = load_config() if cfg is None else cfg
    frequency = frequency or cfg["project"]["frequency"]
    cost_bps = int(cfg["portfolio"]["cost_bps_per_side"])
    ensure_output_dirs(cfg)
    out = resolve(cfg["paths"]["ridge_extension"])
    fig_dir = resolve(cfg["paths"]["ridge_extension_figures"])
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    panel = load_extension_panel(cfg)
    full, s60, s30 = _build_pools(panel, frequency)

    model, grid, fit_info = ext.train_ridge(s60)
    grid.to_csv(out / "validation_grid.csv", index=False)

    test_start = pd.Timestamp("2022-01-01")
    test_end = pd.Timestamp("2026-01-01")
    valid_dates, skipped = [], []
    for date in sorted(full.decision_date.unique()):
        if not (test_start <= date < test_end):
            continue
        block = full.loc[full.decision_date == date]
        if block.label_end_date.isna().any():
            skipped.append((str(date.date()), "unsettled label_end_date"))
            continue
        if not np.isfinite(block.target_forward_return.to_numpy(dtype=float)).all():
            skipped.append((str(date.date()), "non-finite realised return"))
            continue
        valid_dates.append(date)
    if not valid_dates:
        raise RuntimeError("No valid extension test dates")

    selections: Dict[Any, Dict[str, Any]] = {}
    for date in valid_dates:
        a = full.loc[full.decision_date == date]
        b = s60.loc[s60.decision_date == date]
        c = s30.loc[s30.decision_date == date]
        selections[date] = ext.targets_for_date(a, b, c, model)
    assert_nine_strategies(selections)

    ret_lookup = full.set_index(["decision_date", "ticker"]).target_forward_return.to_dict()
    returns, trades, holdings = ext.simulate(selections, ret_lookup, cost_bps)

    # ETFs: one entry fee at the start, held without rebalancing (original rule).
    for etf, column in [("SPY", "spy_return"), ("QQQ", "qqq_return")]:
        for i, date in enumerate(sorted(selections)):
            values = full.loc[full.decision_date == date, column].dropna().unique()
            if len(values) != 1:
                raise ValueError(f"Benchmark {etf} is inconsistent on {date}")
            fee = cost_bps / 10000 if i == 0 else 0.0
            returns.loc[len(returns)] = {
                "strategy": etf,
                "decision_date": date,
                "gross_return": float(values[0]),
                "cost": fee,
                "net_return": float(values[0]) - fee,
                "turnover_two_way": 1.0 if i == 0 else 0.0,
            }

    observed = set(returns.strategy)
    if observed != set(NINE_STRATEGIES):
        raise AssertionError(f"Nine-strategy identifiers changed: {sorted(observed)}")

    summary = ext.performance(returns, frequency)
    total_cost = returns.groupby("strategy").cost.sum().rename("total_cost")
    summary = summary.merge(total_cost, left_on="strategy", right_index=True, how="left")
    wide = returns.pivot(index="decision_date", columns="strategy", values="net_return").sort_index()
    annual = (1.0 + wide).groupby(wide.index.year).prod() - 1.0

    # Public presentation excludes the former R4 sleeve because it is the same
    # economic rule as Main Ridge. The full nine-row records remain on disk for
    # audit and reproducibility.
    public_summary = summary.loc[summary.strategy.isin(PUBLIC_STRATEGIES)].copy()
    public_wide = wide.loc[:, PUBLIC_STRATEGIES].copy()
    public_annual = (1.0 + public_wide).groupby(public_wide.index.year).prod() - 1.0

    # ---- prediction diagnostics on the 60-stock candidate pool --------------
    test60 = s60[
        (s60.decision_date >= test_start)
        & (s60.decision_date < test_end)
        & s60.target_forward_return.notna()
    ].copy()
    test60["prediction"] = model.predict(test60[FEATURES])
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    prediction_metrics = {
        "test_mse": float(mean_squared_error(test60.target_forward_return, test60.prediction)),
        "test_mae": float(mean_absolute_error(test60.target_forward_return, test60.prediction)),
        "test_r2": float(r2_score(test60.target_forward_return, test60.prediction)),
        "test_rows": int(len(test60)),
        "test_dates": int(test60.decision_date.nunique()),
    }
    rank_ic = (
        test60.groupby("decision_date")
        .apply(lambda g: g.prediction.corr(g.target_forward_return, method="spearman"), include_groups=False)
        .rename("rank_ic")
    )
    prediction_metrics["mean_rank_ic"] = float(rank_ic.mean())
    prediction_metrics["rank_ic_positive_share"] = float((rank_ic > 0).mean())

    # ---- persist tables -----------------------------------------------------
    summary.to_csv(out / "strategy_summary.csv", index=False)
    returns.to_csv(out / "period_returns.csv", index=False)
    annual.to_csv(out / "annual_returns.csv")
    public_summary.to_csv(out / "strategy_summary_public.csv", index=False)
    returns.loc[returns.strategy.isin(PUBLIC_STRATEGIES)].to_csv(out / "period_returns_public.csv", index=False)
    public_annual.to_csv(out / "annual_returns_public.csv")
    holdings.to_csv(out / "holdings.csv", index=False)
    trades.to_csv(out / "trades.csv", index=False)
    test60.to_csv(out / "ridge_test60_predictions.csv", index=False)
    rank_ic.to_csv(out / "ridge_rank_ic.csv")
    (out / "prediction_metrics.json").write_text(json.dumps(prediction_metrics, indent=2) + "\n")
    audit_path = resolve(cfg["data"]["extension_panel"]).with_name("extension_screening_audit.csv")
    if audit_path.exists():
        pd.read_csv(audit_path).to_csv(out / "screening_audit.csv", index=False)
    write_json(
        out / "fit_info.json",
        {
            **fit_info,
            "frequency": frequency,
            "cost_bps": cost_bps,
            "test_start": "2022-01-01",
            "test_end_exclusive": "2026-01-01",
            "test_periods": len(valid_dates),
            "skipped_dates": skipped,
            "screening": "60 momentum-only candidates for Ridge training; original 3x3 30-stock pool for R4",
            "strategies_internal": NINE_STRATEGIES,
            "strategies_presented": PUBLIC_STRATEGIES,
            "internal_mapping_note": "R4_RidgeDoubleSort30 is the same economic rule as Main Ridge; retained internally only.",
            "generated_utc": utc_now(),
            "environment": environment_info(),
        },
    )

    # ---- figures ------------------------------------------------------------
    _plot_extension(public_summary, public_wide, public_annual, rank_ic, grid, fig_dir)

    return {
        "summary": summary,
        "public_summary": public_summary,
        "returns": returns,
        "annual": annual,
        "public_annual": public_annual,
        "fit_info": fit_info,
        "grid": grid,
        "prediction_metrics": prediction_metrics,
        "n_periods": len(valid_dates),
        "skipped": skipped,
    }


def _plot_extension(summary, wide, annual, rank_ic, grid, fig_dir: Path) -> None:
    order = summary.sort_values("annualized_return", ascending=False).strategy.tolist()
    colors = plt.cm.tab10(np.linspace(0, 1, len(order)))

    fig, ax = plt.subplots(figsize=(13, 6))
    for color, name in zip(colors, order):
        ax.plot(wide.index, (1.0 + wide[name]).cumprod(), label=name, color=color, linewidth=1.6)
    ax.set_title("Exploratory Ridge extension: cumulative net wealth (2022-2025)")
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("Decision date")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "extension_cumulative_wealth.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    annual.plot(kind="bar", ax=ax, width=0.85)
    ax.set_title("Exploratory Ridge extension: annual net returns")
    ax.set_ylabel("Net return")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(ncol=3, fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "extension_annual_returns.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 5))
    for color, name in zip(colors, order):
        wealth = (1.0 + wide[name]).cumprod()
        ax.plot(wide.index, wealth / wealth.cummax() - 1.0, label=name, color=color, linewidth=1.3)
    ax.set_title("Exploratory Ridge extension: drawdowns")
    ax.set_ylabel("Drawdown")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "extension_drawdowns.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    rank_ic.plot(ax=ax, color="#1f77b4", linewidth=1.2)
    ax.axhline(0, color="grey", linewidth=0.9)
    ax.set_title("Extension Ridge: cross-sectional Rank IC on the 60-candidate pool")
    ax.set_ylabel("Spearman rank IC")
    fig.tight_layout()
    fig.savefig(fig_dir / "ridge_rank_ic.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(grid.alpha, grid.train_mse, "o-", label="Train MSE")
    ax.semilogx(grid.alpha, grid.validation_mse, "o-", label="Validation MSE")
    ax.set_xlabel("Ridge alpha")
    ax.set_ylabel("MSE")
    ax.set_title("Extension Ridge: train vs validation MSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "ridge_validation_curve.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":  # pragma: no cover
    result = run_ridge_extension()
    print(result["summary"].to_string(index=False))
