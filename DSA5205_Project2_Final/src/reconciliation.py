"""Reconcile Main Ridge and R4 RidgeDoubleSort30.

Both strategies apply the same economic rule: the 3x3 momentum-volatility
screen, a Ridge predicted-return Top 10 and equal weights.  They are *not* the
same fitted model.  Main Ridge is trained on the main experiment's 30-candidate
panel (4,260 refit rows); R4 is trained on the extension's 60-candidate
momentum pool (8,520 refit rows).  This module quantifies the consequence and
documents that the two result records must not be merged or counted twice.
"""
from __future__ import annotations

import json
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


def _main_holdings(cfg: Mapping[str, Any]) -> pd.DataFrame:
    path = resolve(cfg["paths"]["oos"]) / "oos_predictions.csv"
    frame = pd.read_csv(path, parse_dates=["decision_date"])
    ridge = frame.loc[frame.model == "ridge"].copy()
    rows = []
    for date, block in ridge.groupby("decision_date"):
        top = block.sort_values(["prediction", "ticker"], ascending=[False, True]).head(10)
        for ticker in top.ticker:
            rows.append({"decision_date": date, "ticker": ticker})
    return pd.DataFrame(rows)


def _r4_holdings(cfg: Mapping[str, Any]) -> pd.DataFrame:
    path = resolve(cfg["paths"]["ridge_extension"]) / "holdings.csv"
    frame = pd.read_csv(path, parse_dates=["decision_date"])
    return frame.loc[frame.strategy == "R4_RidgeDoubleSort30", ["decision_date", "ticker"]].copy()


def run_ridge_r4_reconciliation(cfg: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    cfg = load_config() if cfg is None else cfg
    ensure_output_dirs(cfg)
    ext_dir = resolve(cfg["paths"]["ridge_extension"])
    ext_fig = resolve(cfg["paths"]["ridge_extension_figures"])
    ext_fig.mkdir(parents=True, exist_ok=True)

    main_h = _main_holdings(cfg)
    r4_h = _r4_holdings(cfg)
    dates = sorted(set(main_h.decision_date) & set(r4_h.decision_date))
    overlap_rows = []
    for date in dates:
        a = set(main_h.loc[main_h.decision_date == date, "ticker"])
        b = set(r4_h.loc[r4_h.decision_date == date, "ticker"])
        overlap_rows.append(
            {
                "decision_date": date,
                "n_main": len(a),
                "n_r4": len(b),
                "n_common": len(a & b),
                "holdings_overlap": len(a & b) / 10.0,
                "identical": a == b,
            }
        )
    overlap = pd.DataFrame(overlap_rows)
    overlap.to_csv(ext_dir / "ridge_estimator_holdings_overlap.csv", index=False)

    main_returns = pd.read_csv(resolve(cfg["paths"]["oos"]) / "oos_monthly_returns.csv", parse_dates=["decision_date"])
    ext_returns = pd.read_csv(ext_dir / "period_returns.csv", parse_dates=["decision_date"])
    a = main_returns.loc[main_returns.strategy == "ridge"].set_index("decision_date").net_return
    b = ext_returns.loc[ext_returns.strategy == "R4_RidgeDoubleSort30"].set_index("decision_date").net_return
    aligned = pd.concat([a.rename("main_ridge"), b.rename("r4")], axis=1, join="inner").dropna()

    main_metrics = portfolio_metrics(main_returns, "ridge", frequency=cfg["project"]["frequency"], benchmark="SPY")
    r4_metrics = portfolio_metrics(ext_returns, "R4_RidgeDoubleSort30", frequency=cfg["project"]["frequency"], benchmark="SPY")

    refit_meta = json.loads((resolve(cfg["paths"]["validation"]) / "final_model_metadata.json").read_text())
    fit_info = json.loads((ext_dir / "fit_info.json").read_text())

    summary = {
        "status": "same_economic_rule_different_fitted_model",
        "conclusion": (
            "Main Ridge and R4 apply the same economic trading rule but use different Ridge "
            "estimators trained on different candidate samples. Their performance records are "
            "not numerically interchangeable and must not be merged or counted twice."
        ),
        "main_ridge": {
            "rule": "3x3 screen -> Ridge Top 10 -> equal weight",
            "alpha": refit_meta["models"]["ridge"]["hyperparameters"],
            "refit_rows": refit_meta["models"]["ridge"]["n_training_observations"],
            "candidate_pool_size": 30,
            "training_panel": "main experiment 30-candidate panel",
            "cagr": float(main_metrics["annualized_return"]),
            "sharpe": float(main_metrics["sharpe_ratio"]),
            "max_drawdown": float(main_metrics["max_drawdown"]),
        },
        "r4": {
            "rule": "3x3 screen -> Ridge Top 10 -> equal weight",
            "alpha": fit_info["alpha"],
            "refit_rows": fit_info["refit_rows"],
            "candidate_pool_size": 60,
            "training_panel": "extension 60-candidate momentum pool",
            "cagr": float(r4_metrics["annualized_return"]),
            "sharpe": float(r4_metrics["sharpe_ratio"]),
            "max_drawdown": float(r4_metrics["max_drawdown"]),
        },
        "comparison": {
            "n_dates": int(len(overlap)),
            "mean_holdings_overlap": float(overlap["holdings_overlap"].mean()),
            "min_holdings_overlap": float(overlap["holdings_overlap"].min()),
            "max_holdings_overlap": float(overlap["holdings_overlap"].max()),
            "identical_holding_dates": int(overlap["identical"].sum()),
            "return_correlation": float(aligned["main_ridge"].corr(aligned["r4"])),
            "mean_abs_return_difference": float((aligned["main_ridge"] - aligned["r4"]).abs().mean()),
            "max_abs_return_difference": float((aligned["main_ridge"] - aligned["r4"]).abs().max()),
            "cagr_difference_r4_minus_main": float(
                r4_metrics["annualized_return"] - main_metrics["annualized_return"]
            ),
            "sharpe_difference_r4_minus_main": float(r4_metrics["sharpe_ratio"] - main_metrics["sharpe_ratio"]),
        },
        "reporting_policy": (
            "Main Ridge is the primary result. R4 is labelled as a main-rule reference using the "
            "extension's 60-candidate estimator, shown in the exploratory extension section only, "
            "and excluded from any strategy ranking that already contains Main Ridge."
        ),
        "generated_utc": utc_now(),
    }
    write_json(ext_dir / "ridge_estimator_reconciliation.json", summary)

    # Figure: CAGR/Sharpe bars and holdings-overlap distribution.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    labels = ["Main Ridge\n(30-candidate estimator)", "Extension estimator\n(60-candidate pool)"]
    cagrs = [summary["main_ridge"]["cagr"], summary["r4"]["cagr"]]
    sharpes = [summary["main_ridge"]["sharpe"], summary["r4"]["sharpe"]]
    x = np.arange(2)
    width = 0.36
    bars1 = ax1.bar(x - width / 2, cagrs, width, label="CAGR", color="#12355b")
    ax1.set_ylabel("CAGR")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax1.set_title("(a) Annualised return and Sharpe")
    ax1.set_ylim(0, max(cagrs) * 1.30)
    ax1.grid(axis="y", alpha=0.25)
    ax1b = ax1.twinx()
    bars2 = ax1b.bar(x + width / 2, sharpes, width, label="Sharpe", color="#d97706")
    ax1b.set_ylabel("Sharpe ratio")
    ax1b.set_ylim(0, max(sharpes) * 1.35)
    for i, value in enumerate(cagrs):
        ax1.text(i - width / 2, value + 0.01, f"{value:.1%}", ha="center", fontsize=8)
    for i, value in enumerate(sharpes):
        ax1b.text(i + width / 2, value + 0.02, f"{value:.2f}", ha="center", fontsize=8)
    ax1.set_ylabel("CAGR (navy bars)")
    ax1b.set_ylabel("Sharpe ratio (orange bars)")
    ax2.hist(overlap["holdings_overlap"], bins=np.linspace(0.7, 1.0, 13), color="#2f7d32", edgecolor="white")
    ax2.set_xlabel("Share of the ten holdings held by both strategies")
    ax2.set_ylabel("Number of months")
    ax2.set_title("(b) Holdings overlap, 2022-2025")
    ax2.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig_path = ext_fig / "fig_internal_estimator_reconciliation.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    summary["figure"] = str(fig_path.relative_to(resolve(cfg["paths"]["results"])))
    write_json(ext_dir / "ridge_estimator_reconciliation.json", summary)
    return summary


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(run_ridge_r4_reconciliation(), indent=2))
