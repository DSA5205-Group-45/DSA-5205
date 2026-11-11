"""Final out-of-sample evaluation on 2022-2025.

The selected model is fixed by the frozen selection contract.  Test results are
reported for all four refitted models because the four-way comparison was
planned before the test set was opened; the selection itself is never revisited.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd

from .backtest import combine_model_backtests
from .data_pipeline import (
    chronological_splits,
    ensure_output_dirs,
    load_config,
    load_model_ready,
    resolve,
)
from .final_refit import load_refit_models, load_validation_best_params
from .metrics import metric_definitions, portfolio_metrics, prediction_metrics
from .research_policy import active_contract
from .models import MODEL_LABELS, MODEL_NAMES, params_to_json
from .pipeline import FEATURES
from .utils import environment_info, utc_now, write_json

TARGET = "target_forward_return"


def run_oos(
    cfg: Optional[Mapping[str, Any]] = None,
    frequency: Optional[str] = None,
    contract: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = load_config() if cfg is None else cfg
    frequency = frequency or cfg["project"]["frequency"]
    ensure_output_dirs(cfg)
    contract = dict(contract) if contract is not None else active_contract(cfg)
    selected = contract["selected_model"]

    df = load_model_ready(cfg, frequency)
    splits, split_summary, _ = chronological_splits(df, cfg)
    test = splits["test"].copy()
    refit = splits["refit"]
    estimators = load_refit_models(cfg)
    best = load_validation_best_params(cfg)

    # Benchmark-relative OOS R^2 uses the historical mean of the refit sample:
    # information that was available before 2022-01-01.
    historical_mean = float(refit[TARGET].mean())

    out_dir = resolve(cfg["paths"]["oos"])
    prediction_frames = {}
    prediction_rows = []
    for name in MODEL_NAMES:
        frame = test.copy()
        frame["model"] = name
        frame["model_label"] = MODEL_LABELS[name]
        frame["prediction"] = estimators[name].predict(test[FEATURES])
        frame = frame.sort_values(["decision_date", "prediction", "ticker"], ascending=[True, False, True])
        frame["pred_rank"] = frame.groupby("decision_date").cumcount() + 1
        prediction_frames[name] = frame
        metrics = prediction_metrics(
            frame[TARGET], frame["prediction"], historical_mean, dates=frame["decision_date"]
        )
        prediction_rows.append(
            {
                "model": name,
                "model_label": MODEL_LABELS[name],
                "hyperparameters": params_to_json(best[name]["params"]),
                "is_selected": name == selected,
                "test_rows": int(len(frame)),
                "test_dates": int(frame["decision_date"].nunique()),
                "date_start": str(frame["decision_date"].min().date()),
                "date_end": str(frame["decision_date"].max().date()),
                "historical_mean_baseline": historical_mean,
                **metrics,
            }
        )
    prediction_df = pd.DataFrame(prediction_rows)
    prediction_df.to_csv(out_dir / "oos_prediction_metrics.csv", index=False)

    combined = combine_model_backtests(
        prediction_frames, frequency=frequency, cost_bps=cfg["portfolio"]["cost_bps_per_side"]
    )
    combined.to_csv(out_dir / "oos_monthly_returns.csv", index=False)

    portfolio_rows = []
    for strategy in [*MODEL_NAMES, *cfg["data"]["benchmarks"]]:
        row = portfolio_metrics(combined, strategy, frequency=frequency, benchmark="SPY")
        row["model_label"] = MODEL_LABELS.get(strategy, f"{strategy} buy-and-hold")
        row["is_benchmark"] = strategy in cfg["data"]["benchmarks"]
        row["is_selected"] = strategy == selected
        portfolio_rows.append(row)
    portfolio_df = pd.DataFrame(portfolio_rows)
    for name in MODEL_NAMES:
        portfolio_df.loc[portfolio_df.strategy == name, "hyperparameters"] = params_to_json(best[name]["params"])
    portfolio_df.to_csv(out_dir / "oos_portfolio_metrics.csv", index=False)

    comparison = prediction_df.copy()
    model_port = portfolio_df.loc[portfolio_df.strategy.isin(MODEL_NAMES)].set_index("strategy")
    for column, source in [
        ("oos_sharpe", "sharpe_ratio"),
        ("oos_cagr", "annualized_return"),
        ("oos_total_return", "total_return_net"),
        ("oos_max_drawdown", "max_drawdown"),
        ("oos_calmar", "calmar_ratio"),
        ("oos_avg_turnover", "average_turnover"),
        ("oos_total_cost", "total_transaction_cost"),
        ("oos_beta_to_spy", "beta_to_benchmark"),
        ("oos_alpha_annualized", "alpha_annualized"),
        ("oos_information_ratio", "information_ratio"),
    ]:
        comparison[column] = comparison["model"].map(model_port[source])
    comparison["mse_rank"] = comparison["mse"].rank(method="min", ascending=True).astype(int)
    comparison["sharpe_rank"] = comparison["oos_sharpe"].rank(method="min", ascending=False).astype(int)
    comparison["cagr_rank"] = comparison["oos_cagr"].rank(method="min", ascending=False).astype(int)
    comparison["max_drawdown_rank"] = comparison["oos_max_drawdown"].rank(method="min", ascending=False).astype(int)
    comparison.to_csv(out_dir / "oos_model_comparison.csv", index=False)

    prediction_columns = [
        "model",
        "model_label",
        "decision_date",
        "execution_date",
        "label_end_date",
        "ticker",
        "momentum_group",
        "volatility_group",
        "mom_2_12",
        "prediction",
        "pred_rank",
        TARGET,
        "spy_return",
        "qqq_return",
        "split",
    ]
    predictions_out = pd.concat([prediction_frames[n][prediction_columns] for n in MODEL_NAMES], ignore_index=True)
    predictions_out = predictions_out.sort_values(["model", "decision_date", "pred_rank"]).reset_index(drop=True)
    predictions_out.to_csv(out_dir / "oos_predictions.csv", index=False)

    # ---- validation -> OOS degradation -------------------------------------
    validation_comparison = pd.read_csv(resolve(cfg["paths"]["validation"]) / "validation_model_comparison.csv")
    degradation = comparison[
        [
            "model",
            "mse",
            "rmse",
            "mae",
            "r2_sklearn",
            "r2_oos_trainmean",
            "mean_rank_ic",
            "oos_sharpe",
            "oos_cagr",
            "oos_max_drawdown",
        ]
    ].rename(
        columns={
            "mse": "oos_mse",
            "rmse": "oos_rmse",
            "mae": "oos_mae",
            "r2_sklearn": "oos_r2_sklearn",
            "r2_oos_trainmean": "oos_r2_trainmean",
            "mean_rank_ic": "oos_mean_rank_ic",
        }
    )
    validation_side = validation_comparison[
        [
            "model",
            "validation_mse",
            "validation_rmse",
            "validation_mae",
            "validation_r2_sklearn",
            "validation_r2_oos_trainmean",
            "validation_mean_rank_ic",
            "validation_sharpe",
            "validation_cagr",
            "validation_max_drawdown",
            "best_params",
        ]
    ].rename(
        columns={
            "validation_r2_sklearn": "validation_r2_sklearn",
            "validation_r2_oos_trainmean": "validation_r2_trainmean",
            "validation_mean_rank_ic": "validation_mean_rank_ic",
            "validation_sharpe": "validation_sharpe",
            "validation_cagr": "validation_cagr",
            "validation_max_drawdown": "validation_max_drawdown",
        }
    )
    degradation = validation_side.merge(degradation, on="model", how="inner")
    degradation["mse_change"] = degradation["oos_mse"] - degradation["validation_mse"]
    degradation["mse_ratio"] = degradation["oos_mse"] / degradation["validation_mse"]
    degradation["sharpe_change"] = degradation["oos_sharpe"] - degradation["validation_sharpe"]
    degradation["cagr_change"] = degradation["oos_cagr"] - degradation["validation_cagr"]
    degradation["max_drawdown_change"] = degradation["oos_max_drawdown"] - degradation["validation_max_drawdown"]
    degradation["rank_ic_change"] = degradation["oos_mean_rank_ic"] - degradation["validation_mean_rank_ic"]
    degradation["is_selected"] = degradation["model"] == selected
    degradation.to_csv(out_dir / "validation_to_oos_degradation.csv", index=False)

    # ---- audit: selection must not be revised after seeing test results -----
    audit = {
        "stage": "oos_evaluation",
        "generated_utc": utc_now(),
        "frequency": frequency,
        "selected_model_from_contract": selected,
        "selection_criterion": contract["selection_criterion"],
        "selection_run_id": contract.get("run_id"),
        "selection_timestamp_utc": contract.get("selection_timestamp_utc"),
        "selection_changed_after_test": False,
        "test_period": {
            "start": cfg["data"]["test_start"],
            "end_exclusive": cfg["data"]["test_end_exclusive"],
            "decision_start": str(test.decision_date.min().date()),
            "decision_end": str(test.decision_date.max().date()),
            "label_end_max": str(test.label_end_date.max().date()),
        },
        "test_rows": int(len(test)),
        "test_dates": int(test.decision_date.nunique()),
        "test_data_accessed": True,
        "selected_model_oos_mse_rank": int(
            comparison.loc[comparison.model == selected, "mse_rank"].iloc[0]
        ),
        "selected_model_oos_sharpe_rank": int(
            comparison.loc[comparison.model == selected, "sharpe_rank"].iloc[0]
        ),
        "selected_model_oos_mse": float(comparison.loc[comparison.model == selected, "mse"].iloc[0]),
        "selected_model_oos_sharpe": float(
            comparison.loc[comparison.model == selected, "oos_sharpe"].iloc[0]
        ),
        "descriptive_all_model_comparison": True,
        "note": "All four refitted models are reported on the test period because the comparison was pre-planned. The selected model was not changed after seeing these results.",
        "historical_mean_baseline": historical_mean,
        "random_seed": int(cfg["project"]["random_seed"]),
        "environment": environment_info(),
        "metric_definitions": metric_definitions(),
    }
    write_json(out_dir / "oos_run_metadata.json", audit)
    write_json(out_dir / "final_selection_audit.json", audit)

    return {
        "prediction_metrics": prediction_df,
        "portfolio_metrics": portfolio_df,
        "comparison": comparison,
        "predictions": predictions_out,
        "monthly_returns": combined,
        "degradation": degradation,
        "metadata": audit,
        "selected_model": selected,
    }


if __name__ == "__main__":  # pragma: no cover
    result = run_oos()
    print(result["comparison"][["model", "mse", "oos_sharpe", "oos_cagr"]].to_string(index=False))
