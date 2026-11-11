"""LEGACY / SUPERSEDED auxiliary automatic model selection.

This module implements the earlier engineering convenience that ranked the four
models by ``lowest_validation_mse`` and selected LASSO.  It is **not** used by
the active pipeline, notebook or report.  The research team's primary model is
Ridge, designated in :mod:`src.research_policy`.  The module is retained only
to reproduce and audit the historical run, and its archived outputs live under
``results/validation/archive/superseded_automatic_selection/``.

Original documentation follows.

Explicit, configurable and auditable final-model selection.

The selection rule is fixed from the validation comparison alone.  The test
period is never read here.  A selection can only be frozen when the caller
passes ``approve=True`` (the CLI requires ``--approve``; the notebook exposes a
prominent ``APPROVE_FINAL_SELECTION`` flag).  Without approval the stage raises
:class:`SelectionNotApproved` and the pipeline stops before OOS evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd

from .data_pipeline import ensure_output_dirs, load_config, resolve
from .models import MODEL_NAMES, params_to_json
from .utils import environment_info, stable_hash, utc_now, write_json


class SelectionNotApproved(RuntimeError):
    """Raised when OOS evaluation is attempted without a frozen selection."""


SELECTION_RULES = {
    "lowest_validation_mse": {
        "sort_column": "validation_mse",
        "ascending": True,
        "description": "Lowest out-of-sample validation MSE (statistical accuracy). This is the pre-specified default.",
    },
    "highest_validation_sharpe": {
        "sort_column": "validation_sharpe",
        "ascending": False,
        "description": "Highest post-cost validation Sharpe ratio (investment performance). Research alternative.",
    },
}


def _comparison_path(cfg: Mapping[str, Any]) -> Path:
    return resolve(cfg["paths"]["validation"]) / "validation_model_comparison.csv"


def load_validation_comparison(cfg: Mapping[str, Any]) -> pd.DataFrame:
    path = _comparison_path(cfg)
    if not path.exists():
        raise FileNotFoundError(
            f"Validation comparison not found at {path}. Run the validation stage first."
        )
    frame = pd.read_csv(path)
    required = {"model", "best_params", "validation_mse", "validation_sharpe"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Validation comparison is missing columns: {sorted(missing)}")
    if set(frame["model"]) != set(MODEL_NAMES):
        raise ValueError("Validation comparison must contain exactly the four candidate models")
    return frame


def _fingerprint(comparison: pd.DataFrame, criterion: str) -> str:
    payload = {
        "criterion": criterion,
        "models": [
            {
                "model": row.model,
                "params": json.loads(row.best_params) if isinstance(row.best_params, str) else row.best_params,
                "validation_mse": float(row.validation_mse),
                "validation_sharpe": float(row.validation_sharpe),
            }
            for row in comparison.sort_values("model").itertuples()
        ],
    }
    return stable_hash(payload)


def _ranked_comparison(comparison: pd.DataFrame, criterion: str) -> pd.DataFrame:
    rule = SELECTION_RULES[criterion]
    frame = comparison.copy()
    frame["mse_rank"] = frame["validation_mse"].rank(method="min", ascending=True).astype(int)
    frame["sharpe_rank"] = frame["validation_sharpe"].rank(method="min", ascending=False).astype(int)
    order = frame.sort_values([rule["sort_column"], "model"], ascending=[rule["ascending"], True])
    frame["selection_rank"] = order["model"].rank(method="first").reindex(frame.index).astype(int)
    frame["selected"] = frame["model"] == order.iloc[0]["model"]
    frame["selection_criterion"] = criterion
    frame["is_primary_model"] = frame["model"] == "ridge"
    return frame


def select_final_model(
    cfg: Optional[Mapping[str, Any]] = None,
    criterion: Optional[str] = None,
    approve: bool = False,
    note: str = "",
) -> Dict[str, Any]:
    """Freeze the final model. Returns the selection contract dictionary.

    Re-running with the same validation fingerprint and criterion is
    idempotent: the original ``selection_timestamp_utc`` is preserved and only
    ``last_verified_utc`` is refreshed.  This keeps the audit trail honest when
    the notebook re-executes after the OOS stage.
    """
    cfg = load_config() if cfg is None else cfg
    ensure_output_dirs(cfg)
    criterion = criterion or cfg["selection"]["criterion"]
    if criterion not in cfg["selection"]["allowed_criteria"]:
        raise ValueError(
            f"Unknown selection criterion {criterion!r}; allowed: {cfg['selection']['allowed_criteria']}"
        )
    comparison = load_validation_comparison(cfg)
    ranked = _ranked_comparison(comparison, criterion)
    winner = ranked.loc[ranked["selected"]].iloc[0]
    mse_winner = ranked.loc[ranked["validation_mse"].idxmin(), "model"]
    sharpe_winner = ranked.loc[ranked["validation_sharpe"].idxmax(), "model"]
    fingerprint = _fingerprint(comparison, criterion)

    out_dir = resolve(cfg["paths"]["validation"])
    summary_path = out_dir / "model_selection_summary.csv"
    summary_columns = [
        "model",
        "best_params",
        "validation_mse",
        "validation_sharpe",
        "validation_cagr",
        "validation_max_drawdown",
        "mse_rank",
        "sharpe_rank",
        "selection_rank",
        "selected",
        "selection_criterion",
        "is_primary_model",
    ]
    for column in summary_columns:
        if column not in ranked.columns:
            ranked[column] = np.nan
    ranked[summary_columns].sort_values("selection_rank").to_csv(summary_path, index=False)

    contract_path = out_dir / "selection_contract.json"
    now = utc_now()
    existing = None
    if contract_path.exists():
        try:
            existing = json.loads(contract_path.read_text())
        except json.JSONDecodeError:
            existing = None

    if existing and existing.get("validation_fingerprint") == fingerprint:
        if existing.get("test_used_for_selection") is not False:
            raise RuntimeError("Existing selection contract is invalid: test data was used")
        existing["last_verified_utc"] = now
        existing["status"] = "frozen"
        existing["last_verification"] = "existing_contract_reused"
        write_json(contract_path, existing)
        return existing

    if not approve:
        pending = {
            "status": "awaiting_explicit_approval",
            "selection_criterion": criterion,
            "proposed_model": winner["model"],
            "proposed_hyperparameters": json.loads(winner["best_params"]),
            "validation_mse_winner": mse_winner,
            "validation_sharpe_winner": sharpe_winner,
            "test_data_accessed": False,
            "generated_utc": now,
            "instruction": "Re-run with approve=True (CLI: --approve) to freeze the selection before OOS evaluation.",
        }
        write_json(out_dir / "model_selection_pending.json", pending)
        raise SelectionNotApproved(
            "Final model selection has not been approved. "
            "Inspect validation/model_selection_summary.csv and re-run with --approve."
        )

    params = json.loads(winner["best_params"])
    contract = {
        "schema_version": "1.0",
        "status": "frozen",
        "run_id": f"sel-{fingerprint[:16]}",
        "selection_timestamp_utc": now,
        "last_verified_utc": now,
        "primary_model": cfg["selection"]["primary_model"],
        "selection_criterion": criterion,
        "selection_rule_description": SELECTION_RULES[criterion]["description"],
        "selected_model": winner["model"],
        "selected_hyperparameters": params,
        "selected_model_validation_metrics": {
            "mse": float(winner["validation_mse"]),
            "rmse": float(winner.get("validation_rmse", np.nan)),
            "mae": float(winner.get("validation_mae", np.nan)),
            "r2_sklearn": float(winner.get("validation_r2_sklearn", np.nan)),
            "r2_oos_trainmean": float(winner.get("validation_r2_oos_trainmean", np.nan)),
            "sharpe_ratio_rf0": float(winner["validation_sharpe"]),
            "cagr": float(winner.get("validation_cagr", np.nan)),
            "max_drawdown": float(winner.get("validation_max_drawdown", np.nan)),
        },
        "validation_period": {
            "start": cfg["data"]["validation_start"],
            "end_exclusive": cfg["data"]["validation_end_exclusive"],
            "label_embargo": cfg["data"]["label_embargo"],
        },
        "validation_mse_winner": mse_winner,
        "validation_sharpe_winner": sharpe_winner,
        "candidate_models": list(MODEL_NAMES),
        "candidate_hyperparameter_configurations": int(
            comparison["grid_candidates"].sum() if "grid_candidates" in comparison else np.nan
        ),
        "test_used_for_selection": False,
        "test_data_accessed": False,
        "approval": {
            "approved": True,
            "approved_by": "explicit caller approval (CLI --approve or notebook APPROVE_FINAL_SELECTION)",
            "note": note,
        },
        "validation_fingerprint": fingerprint,
        "config_hash": stable_hash({k: v for k, v in cfg.items() if not k.startswith("_")}),
        "environment": environment_info(),
    }
    write_json(contract_path, contract)
    return contract


def require_selection(cfg: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Load and validate the frozen selection contract before any OOS work."""
    cfg = load_config() if cfg is None else cfg
    path = resolve(cfg["paths"]["validation"]) / "selection_contract.json"
    if not path.exists():
        raise SelectionNotApproved(
            "No frozen selection contract. OOS evaluation is blocked until the "
            "selection stage is run with explicit approval (--approve)."
        )
    contract = json.loads(path.read_text())
    if contract.get("status") != "frozen":
        raise SelectionNotApproved("Selection contract is not in the 'frozen' state")
    if contract.get("test_used_for_selection") is not False:
        raise RuntimeError("Selection contract violates the test-independence guarantee")
    if contract.get("test_data_accessed") is not False:
        raise RuntimeError("Selection contract reports that test data was accessed")
    if contract.get("selected_model") not in MODEL_NAMES:
        raise RuntimeError(f"Unknown selected model: {contract.get('selected_model')!r}")
    comparison = load_validation_comparison(cfg)
    fingerprint = _fingerprint(comparison, contract["selection_criterion"])
    if fingerprint != contract.get("validation_fingerprint"):
        raise RuntimeError(
            "Validation results changed after the selection was frozen. "
            "Re-run the selection stage and re-approve before evaluating OOS."
        )
    return contract


def selection_summary_table(cfg: Optional[Mapping[str, Any]] = None) -> pd.DataFrame:
    cfg = load_config() if cfg is None else cfg
    path = resolve(cfg["paths"]["validation"]) / "model_selection_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Selection summary not found at {path}")
    return pd.read_csv(path)
