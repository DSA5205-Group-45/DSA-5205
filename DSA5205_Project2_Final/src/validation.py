"""Training on 2010-2019, hyperparameter tuning and evaluation on 2020-2021.

The validation stage is fully separated from the test stage:

* hyperparameters are ranked **only** by validation MSE (the configured
  default) or validation Sharpe (the documented alternative);
* validation predictions come from a model fitted on the training slice only,
  so ``StandardScaler`` means/variances and tree structure never see 2020-2021
  rows during fitting;
* the test slice is never loaded by this module.
"""
from __future__ import annotations

import json
import warnings
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from .backtest import combine_model_backtests, run_single_strategy_backtest
from .data_pipeline import (
    chronological_splits,
    data_quality_report,
    ensure_output_dirs,
    load_config,
    load_model_ready,
    resolve,
)
from .metrics import (
    metric_definitions,
    portfolio_metrics,
    prediction_metrics,
)
from .models import (
    MODEL_LABELS,
    MODEL_NAMES,
    build_model,
    convergence_info,
    parameter_grid,
    params_to_json,
    select_best_config,
)
from .pipeline import FEATURES
from .utils import environment_info, stable_hash, utc_now, write_json

TARGET = "target_forward_return"


def _format_params(params: Mapping[str, Any], columns: Optional[list] = None) -> Dict[str, Any]:
    out = {f"param_{k}": v for k, v in params.items()}
    if columns is not None:
        for key in columns:
            out.setdefault(f"param_{key}", np.nan)
    return out


def tune_all_models(
    splits: Mapping[str, pd.DataFrame],
    cfg: Mapping[str, Any],
    criterion: str = "lowest_validation_mse",
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]], Dict[str, pd.DataFrame]]:
    """Fit every grid configuration on train, score it on validation.

    Returns ``(search_table, best_configs, validation_predictions)`` where
    ``validation_predictions`` maps model name to a frame of validation rows
    with a ``prediction`` column produced by the selected configuration.
    """
    train = splits["train"]
    validation = splits["validation"]
    seed = int(cfg["project"]["random_seed"])
    train_mean = float(train[TARGET].mean())

    if train.empty or validation.empty:
        raise ValueError("Training and validation slices must both be non-empty")

    search_rows: list[Dict[str, Any]] = []
    best_configs: Dict[str, Dict[str, Any]] = {}
    prediction_frames: Dict[str, pd.DataFrame] = {}

    for name in MODEL_NAMES:
        grid = parameter_grid(name, cfg["models"][name])
        candidates = []
        for index, params in enumerate(grid):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimator = build_model(name, params, seed)
                estimator.fit(train[FEATURES], train[TARGET])
                train_pred = estimator.predict(train[FEATURES])
                validation_pred = estimator.predict(validation[FEATURES])
            train_metrics = prediction_metrics(
                train[TARGET], train_pred, train_mean, dates=train["decision_date"]
            )
            val_metrics = prediction_metrics(
                validation[TARGET], validation_pred, train_mean, dates=validation["decision_date"]
            )
            convergence = convergence_info(name, estimator)
            row = {
                "model": name,
                "model_label": MODEL_LABELS[name],
                "config_index": index,
                "params": params_to_json(params),
                **_format_params(
                    params,
                    sorted({k for g in grid for k in g}),
                ),
                "n_features": len(FEATURES),
                "train_rows": int(len(train)),
                "validation_rows": int(len(validation)),
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"validation_{k}": v for k, v in val_metrics.items()},
                **{f"convergence_{k}": v for k, v in convergence.items()},
                "selected": False,
            }
            search_rows.append(row)
            candidates.append((params, estimator, validation_pred, row))

        search = pd.DataFrame([c[3] for c in candidates])
        best_row = select_best_config(search, criterion)
        best_index = int(best_row["config_index"])
        best_params = dict(grid[best_index])
        for row in search_rows:
            if row["model"] == name and row["config_index"] == best_index:
                row["selected"] = True
        best_configs[name] = {
            "params": best_params,
            "config_index": best_index,
            "n_candidates": len(grid),
            "train_mse": float(best_row["train_mse"]),
            "validation_mse": float(best_row["validation_mse"]),
            "validation_sharpe": np.nan,
            "selection_criterion": criterion,
        }
        frame = validation.copy()
        frame["model"] = name
        frame["model_label"] = MODEL_LABELS[name]
        frame["prediction"] = candidates[best_index][2]
        frame["pred_rank"] = (
            frame.sort_values(["decision_date", "prediction", "ticker"], ascending=[True, False, True])
            .groupby("decision_date")
            .cumcount()
            .add(1)
            .reindex(frame.index)
        )
        prediction_frames[name] = frame

    return pd.DataFrame(search_rows).sort_values(["model", "config_index"]).reset_index(drop=True), best_configs, prediction_frames


def _rank(series: pd.Series, ascending: bool) -> pd.Series:
    return series.rank(method="min", ascending=ascending).astype(int)


def run_validation(
    cfg: Optional[Mapping[str, Any]] = None,
    frequency: Optional[str] = None,
    criterion: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute the complete validation comparison and persist every table."""
    cfg = load_config() if cfg is None else cfg
    frequency = frequency or cfg["project"]["frequency"]
    criterion = criterion or cfg["selection"]["criterion"]
    if criterion not in cfg["selection"]["allowed_criteria"]:
        raise ValueError(f"Unknown selection criterion: {criterion!r}")
    ensure_output_dirs(cfg)
    out_dir = resolve(cfg["paths"]["validation"])

    df = load_model_ready(cfg, frequency)
    splits, split_summary, excluded = chronological_splits(df, cfg)
    quality = data_quality_report(df, splits)
    write_json(out_dir / "data_quality.json", quality)
    split_summary.to_csv(out_dir / "split_summary.csv", index=False)
    excluded.to_csv(out_dir / "label_embargo_excluded.csv", index=False)
    if not quality["pass"]:
        raise RuntimeError(f"Data quality checks failed: {quality['issues']}")

    search, best_configs, predictions = tune_all_models(splits, cfg, criterion)
    search.to_csv(out_dir / "validation_hyperparameter_search.csv", index=False)

    # ---- prediction metrics -------------------------------------------------
    prediction_rows = []
    for name in MODEL_NAMES:
        frame = predictions[name]
        metrics = prediction_metrics(
            frame[TARGET], frame["prediction"], float(splits["train"][TARGET].mean()), dates=frame["decision_date"]
        )
        prediction_rows.append(
            {
                "model": name,
                "model_label": MODEL_LABELS[name],
                "best_params": params_to_json(best_configs[name]["params"]),
                "grid_candidates": best_configs[name]["n_candidates"],
                "train_rows": int(len(splits["train"])),
                "validation_rows": int(len(frame)),
                "validation_dates": int(frame["decision_date"].nunique()),
                "date_start": str(frame["decision_date"].min().date()),
                "date_end": str(frame["decision_date"].max().date()),
                **metrics,
            }
        )
    prediction_df = pd.DataFrame(prediction_rows)
    prediction_df.to_csv(out_dir / "validation_prediction_metrics.csv", index=False)

    # ---- portfolio backtests (original rules for every model) ---------------
    combined = combine_model_backtests(predictions, frequency=frequency, cost_bps=cfg["portfolio"]["cost_bps_per_side"])
    combined.to_csv(out_dir / "validation_monthly_returns.csv", index=False)

    portfolio_rows = []
    for strategy in [*MODEL_NAMES, *cfg["data"]["benchmarks"]]:
        row = portfolio_metrics(combined, strategy, frequency=frequency, benchmark="SPY")
        row["model_label"] = MODEL_LABELS.get(strategy, f"{strategy} buy-and-hold")
        row["is_benchmark"] = strategy in cfg["data"]["benchmarks"]
        portfolio_rows.append(row)
    portfolio_df = pd.DataFrame(portfolio_rows)

    # Attach the realised validation Sharpe to the best-config record.
    sharpe_lookup = portfolio_df.set_index("strategy")["sharpe_ratio"].to_dict()
    for name in MODEL_NAMES:
        best_configs[name]["validation_sharpe"] = float(sharpe_lookup.get(name, np.nan))

    for name in MODEL_NAMES:
        portfolio_df.loc[portfolio_df.strategy == name, "best_params"] = params_to_json(
            best_configs[name]["params"]
        )
    portfolio_df.to_csv(out_dir / "validation_portfolio_metrics.csv", index=False)

    # ---- comparison table ---------------------------------------------------
    model_port = portfolio_df.loc[portfolio_df.strategy.isin(MODEL_NAMES)].set_index("strategy")
    comparison = prediction_df.copy()
    # Explicit validation_* aliases keep the comparison table self-documenting
    # and distinguish validation statistics from the later OOS statistics.
    comparison["validation_mse"] = comparison["mse"]
    comparison["validation_rmse"] = comparison["rmse"]
    comparison["validation_mae"] = comparison["mae"]
    comparison["validation_r2_sklearn"] = comparison["r2_sklearn"]
    comparison["validation_r2_oos_trainmean"] = comparison["r2_oos_trainmean"]
    comparison["validation_mean_rank_ic"] = comparison["mean_rank_ic"]
    for column, source in [
        ("validation_sharpe", "sharpe_ratio"),
        ("validation_cagr", "annualized_return"),
        ("validation_total_return", "total_return_net"),
        ("validation_max_drawdown", "max_drawdown"),
        ("validation_calmar", "calmar_ratio"),
        ("validation_avg_turnover", "average_turnover"),
        ("validation_total_cost", "total_transaction_cost"),
    ]:
        comparison[column] = comparison["model"].map(model_port[source])
    comparison["mse_rank"] = _rank(comparison["validation_mse"], ascending=True)
    comparison["sharpe_rank"] = _rank(comparison["validation_sharpe"], ascending=False)
    comparison["cagr_rank"] = _rank(comparison["validation_cagr"], ascending=False)
    comparison["max_drawdown_rank"] = _rank(comparison["validation_max_drawdown"], ascending=False)
    comparison["is_primary_model"] = comparison["model"] == cfg["selection"]["primary_model"]
    comparison["is_validation_mse_winner"] = comparison["model"] == comparison.loc[comparison["validation_mse"].idxmin(), "model"]
    comparison["is_validation_sharpe_winner"] = comparison["model"] == comparison.loc[comparison["validation_sharpe"].idxmax(), "model"]
    comparison = comparison.sort_values("validation_mse").reset_index(drop=True)
    comparison.to_csv(out_dir / "validation_model_comparison.csv", index=False)

    # ---- predictions --------------------------------------------------------
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
    predictions_out = pd.concat([predictions[n][prediction_columns] for n in MODEL_NAMES], ignore_index=True)
    predictions_out = predictions_out.sort_values(["model", "decision_date", "pred_rank"]).reset_index(drop=True)
    predictions_out.to_csv(out_dir / "validation_predictions.csv", index=False)

    write_json(
        out_dir / "validation_best_params.json",
        {
            "criterion": criterion,
            "frequency": frequency,
            "models": {
                name: {
                    "params": best_configs[name]["params"],
                    "config_index": best_configs[name]["config_index"],
                    "n_candidates": best_configs[name]["n_candidates"],
                    "validation_mse": best_configs[name]["validation_mse"],
                }
                for name in MODEL_NAMES
            },
        },
    )

    run_metadata = {
        "stage": "validation",
        "generated_utc": utc_now(),
        "frequency": frequency,
        "selection_criterion": criterion,
        "primary_model": cfg["selection"]["primary_model"],
        "target": TARGET,
        "features": FEATURES,
        "n_features": len(FEATURES),
        "train_rows": int(len(splits["train"])),
        "train_dates": int(splits["train"].decision_date.nunique()),
        "validation_rows": int(len(splits["validation"])),
        "validation_dates": int(splits["validation"].decision_date.nunique()),
        "validation_decision_start": str(splits["validation"].decision_date.min().date()),
        "validation_decision_end": str(splits["validation"].decision_date.max().date()),
        "validation_label_end_max": str(splits["validation"].label_end_date.max().date()),
        "test_rows_loaded": 0,
        "test_data_accessed": False,
        "label_embargo": cfg["data"]["label_embargo"],
        "grid_candidates": {name: best_configs[name]["n_candidates"] for name in MODEL_NAMES},
        "grid_candidates_total": int(sum(best_configs[name]["n_candidates"] for name in MODEL_NAMES)),
        "random_seed": int(cfg["project"]["random_seed"]),
        "config_hash": stable_hash({k: v for k, v in cfg.items() if not k.startswith("_")}),
        "environment": environment_info(),
        "metric_definitions": metric_definitions(),
    }
    write_json(out_dir / "validation_run_metadata.json", run_metadata)

    return {
        "config": cfg,
        "split_summary": split_summary,
        "excluded": excluded,
        "search": search,
        "best_configs": best_configs,
        "prediction_metrics": prediction_df,
        "portfolio_metrics": portfolio_df,
        "comparison": comparison,
        "predictions": predictions_out,
        "monthly_returns": combined,
        "metadata": run_metadata,
    }


if __name__ == "__main__":  # pragma: no cover
    result = run_validation()
    print(result["comparison"][["model", "best_params", "validation_mse", "validation_sharpe"]].to_string(index=False))
