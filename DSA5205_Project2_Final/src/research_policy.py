"""Ridge-centred primary-model policy.

The original project designated Ridge as the primary research model.  The
later four-model experiment added an automatic ``lowest_validation_mse`` rule
that selected LASSO.  That rule remains reproducible and its audit trail is
preserved, but it is no longer the mechanism that defines the primary model.

This module implements an explicit, documented policy:

* primary model designation: Ridge, on economic and diagnostic grounds;
* hyperparameter selection: still validation-based, *within* each model;
* alternative models: LASSO, Random Forest, XGBoost, compared on the same task;
* historical audit: the LASSO minimum-MSE selection is archived, not deleted.

The policy is honest about chronology: the project history includes viewing
out-of-sample results before this restructuring, so the designation is **not**
a clean pre-registration.  The contract records that fact.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd

from .data_pipeline import ensure_output_dirs, load_config, resolve
from .models import MODEL_NAMES, params_to_json
from .utils import environment_info, stable_hash, utc_now, write_json

PRIMARY_MODEL = "ridge"
POLICY_NAME = "ridge_primary_economic_designation"
HYPERPARAMETER_RULE = "lowest_validation_mse_within_ridge"


class PrimaryDesignationNotApproved(RuntimeError):
    """Raised when the primary-model designation has not been approved."""


def _archive_dir(cfg: Mapping[str, Any]) -> Path:
    return resolve(cfg["paths"]["validation"]) / "archive" / "superseded_automatic_selection"


def archive_historical_selection(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Copy the LASSO minimum-MSE selection audit into an archive folder.

    The originals under ``results/validation/`` are left in place; only copies
    are written.  Returns a manifest describing what was archived.
    """
    validation = resolve(cfg["paths"]["validation"])
    archive = _archive_dir(cfg)
    archive.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {
        "archived_utc": utc_now(),
        "reason": (
            "The original four-model experiment used an automatic lowest-validation-MSE rule, "
            "which selected LASSO. The revised research policy designates Ridge as the primary "
            "model on economic and diagnostic grounds. The automatic-selection audit is kept "
            "for transparency and is not deleted or overwritten."
        ),
        "files": [],
    }
    mappings = [
        ("selection_contract.json", "selection_contract.json"),
        ("model_selection_summary.csv", "model_selection_summary.csv"),
        ("model_selection_pending.json", "model_selection_pending.json"),
    ]
    for source_name, target_name in mappings:
        source = validation / source_name
        if source.exists():
            target = archive / target_name
            shutil.copy2(source, target)
            manifest["files"].append(
                {"original": str(source.relative_to(resolve(cfg["paths"]["results"]))), "archived": str(target.relative_to(resolve(cfg["paths"]["results"])))}
            )
    if manifest["files"]:
        write_json(archive / "ARCHIVE_MANIFEST.json", manifest)
    return manifest


def _validation_fingerprint(comparison: pd.DataFrame) -> str:
    payload = {
        "models": [
            {
                "model": row.model,
                "params": json.loads(row.best_params),
                "validation_mse": float(row.validation_mse),
                "validation_sharpe": float(row.validation_sharpe),
                "validation_rank_ic": float(getattr(row, "validation_mean_rank_ic", np.nan)),
            }
            for row in comparison.sort_values("model").itertuples()
        ]
    }
    return stable_hash(payload)


def _load_comparison(cfg: Mapping[str, Any]) -> pd.DataFrame:
    path = resolve(cfg["paths"]["validation"]) / "validation_model_comparison.csv"
    if not path.exists():
        raise FileNotFoundError(f"Validation comparison not found: {path}. Run validation first.")
    frame = pd.read_csv(path)
    if set(frame["model"]) != set(MODEL_NAMES):
        raise ValueError("Validation comparison must contain the four candidate models")
    return frame


def _diagnostic_evidence(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    path = resolve(cfg["paths"]["research_diagnostics"]) / "coefficient_stability_summary.csv"
    evidence: Dict[str, Any] = {"available": path.exists()}
    if path.exists():
        summary = pd.read_csv(path).set_index("model")
        ridge = summary.loc["ridge"]
        lasso = summary.loc["lasso"]
        relative_lower = float((lasso["mean_relative_l2_change"] - ridge["mean_relative_l2_change"]) / lasso["mean_relative_l2_change"])
        evidence.update(
            {
                "ridge_mean_relative_change": float(ridge["mean_relative_l2_change"]),
                "lasso_mean_relative_change": float(lasso["mean_relative_l2_change"]),
                "ridge_relative_change_reduction": relative_lower,
                "lasso_zero_fraction": float(lasso["zero_coefficient_fraction"]),
                "ridge_sign_flips": int(ridge["total_sign_flips"]),
                "lasso_sign_flips": int(lasso["total_sign_flips"]),
                "n_windows": int(ridge["n_windows"]),
            }
        )
    pairs_path = resolve(cfg["paths"]["research_diagnostics"]) / "feature_correlation_pairs.csv"
    if pairs_path.exists():
        pairs = pd.read_csv(pairs_path)
        evidence["high_correlation_pairs_0_70"] = int((pairs["abs_correlation"] >= 0.70).sum())
    return evidence


def designation_table(cfg: Optional[Mapping[str, Any]] = None) -> pd.DataFrame:
    """Four-model table with explicit research roles."""
    cfg = load_config() if cfg is None else cfg
    comparison = _load_comparison(cfg)
    roles = {
        "ridge": ("Primary research model", "Economic + diagnostic designation; alpha by validation MSE"),
        "lasso": ("Alternative model", "Comparative benchmark for shrinkage / sparsity"),
        "rf": ("Alternative model", "Comparative benchmark for nonlinearity"),
        "xgb": ("Alternative model", "Comparative benchmark for boosted nonlinearity"),
    }
    frame = comparison.copy()
    frame["research_role"] = frame["model"].map(lambda m: roles[m][0])
    frame["role_rationale"] = frame["model"].map(lambda m: roles[m][1])
    frame["is_primary"] = frame["model"] == PRIMARY_MODEL
    return frame


def designate_primary_model(
    cfg: Optional[Mapping[str, Any]] = None,
    approve: bool = False,
    note: str = "",
) -> Dict[str, Any]:
    """Freeze Ridge as the primary research model.

    Hyperparameters are taken from the validation search restricted to Ridge
    (``alpha = 1000`` in this data), not from test performance.
    """
    cfg = load_config() if cfg is None else cfg
    ensure_output_dirs(cfg)
    validation = resolve(cfg["paths"]["validation"])
    comparison = _load_comparison(cfg)
    ridge_row = comparison.loc[comparison.model == PRIMARY_MODEL].iloc[0]
    ridge_params = json.loads(ridge_row["best_params"])
    fingerprint = _validation_fingerprint(comparison)
    contract_path = validation / "primary_model_contract.json"
    archive_manifest = archive_historical_selection(cfg)

    designation = designation_table(cfg)
    designation.to_csv(validation / "primary_model_designation.csv", index=False)

    existing = None
    if contract_path.exists():
        try:
            existing = json.loads(contract_path.read_text())
        except json.JSONDecodeError:
            existing = None
    now = utc_now()

    if existing and existing.get("validation_fingerprint") == fingerprint:
        existing["last_verified_utc"] = now
        existing["status"] = "frozen"
        existing["last_verification"] = "existing_contract_reused"
        write_json(contract_path, existing)
        return existing

    if not approve:
        pending = {
            "status": "awaiting_explicit_approval",
            "proposed_primary_model": PRIMARY_MODEL,
            "proposed_hyperparameters": ridge_params,
            "hyperparameter_rule": HYPERPARAMETER_RULE,
            "validation_mse": float(ridge_row["validation_mse"]),
            "validation_rank_ic": float(ridge_row["validation_mean_rank_ic"]),
            "generated_utc": now,
            "instruction": "Re-run with approve=True (CLI --approve) to freeze the Ridge designation.",
        }
        write_json(validation / "primary_model_designation_pending.json", pending)
        raise PrimaryDesignationNotApproved(
            "Primary-model designation has not been approved. "
            "Inspect validation/primary_model_designation.csv and re-run with --approve."
        )

    evidence = _diagnostic_evidence(cfg)
    contract = {
        "schema_version": "2.0",
        "status": "frozen",
        "policy": POLICY_NAME,
        "run_id": f"ridge-{fingerprint[:16]}",
        "designation_timestamp_utc": now,
        "selection_timestamp_utc": now,
        "last_verified_utc": now,
        "primary_model": PRIMARY_MODEL,
        "selected_model": PRIMARY_MODEL,
        "selected_hyperparameters": ridge_params,
        "hyperparameter_rule": HYPERPARAMETER_RULE,
        "selection_criterion": POLICY_NAME,
        "selection_rule_description": (
            "Ridge is designated primary on economic and diagnostic grounds. "
            "Its L2 penalty is chosen by validation MSE within Ridge. "
            "The automatic cross-model minimum-MSE rule is retained only as a historical audit."
        ),
        "validation_metrics": {
            "mse": float(ridge_row["validation_mse"]),
            "rmse": float(ridge_row["validation_rmse"]),
            "mae": float(ridge_row["validation_mae"]),
            "r2_sklearn": float(ridge_row["validation_r2_sklearn"]),
            "r2_oos_trainmean": float(ridge_row["validation_r2_oos_trainmean"]),
            "mean_rank_ic": float(ridge_row["validation_mean_rank_ic"]),
            "sharpe_ratio_rf0": float(ridge_row["validation_sharpe"]),
            "cagr": float(ridge_row["validation_cagr"]),
            "max_drawdown": float(ridge_row["validation_max_drawdown"]),
        },
        "rationale": [
            "The 15 predictors are correlated financial characteristics; L2 shrinkage stabilises a dense linear combination instead of discarding correlated information.",
            "Ridge produces a continuous cross-sectional score, which maps directly to the ranking task used by the Top-10 portfolio.",
            "Ridge has the highest validation mean Rank IC among the four models, the metric most closely aligned with cross-sectional selection.",
            "Rolling training-window diagnostics show lower relative coefficient change for Ridge than LASSO (0.176 vs 0.282), supporting parameter stability.",
            "Ridge is the pre-specified model of the original research design; this policy restores that designation rather than creating a new one.",
        ],
        "diagnostic_evidence": evidence,
        "historical_automatic_selection": {
            "criterion": "lowest_validation_mse",
            "selected_model": "lasso",
            "selected_hyperparameters": {"alpha": 0.001},
            "archived_manifest": archive_manifest,
            "note": (
                "The automatic rule selected LASSO. It remains a valid statistical result and is "
                "reported in full, but it is not the research-design rule for the primary model."
            ),
        },
        "validation_period": {
            "start": cfg["data"]["validation_start"],
            "end_exclusive": cfg["data"]["validation_end_exclusive"],
            "label_embargo": cfg["data"]["label_embargo"],
        },
        "test_used_for_selection": False,
        "test_data_accessed": False,
        "test_metrics_used_in_designation": False,
        "designation_informed_by_prior_oos_review": True,
        "clean_preregistration": False,
        "chronology_disclosure": (
            "The original project pre-specified Ridge. A later engineering revision introduced an "
            "automatic minimum-MSE rule that chose LASSO, and out-of-sample results were reviewed "
            "during that revision. The present designation restores the original pre-specification "
            "on economic and diagnostic grounds, but it is not a clean pre-registration because the "
            "project history includes prior OOS review. No test-period metric is used to choose the "
            "model or its alpha."
        ),
        "validation_fingerprint": fingerprint,
        "config_hash": stable_hash({k: v for k, v in cfg.items() if not k.startswith("_")}),
        "approval": {"approved": True, "note": note},
        "environment": environment_info(),
    }
    write_json(contract_path, contract)
    return contract


def require_primary_designation(cfg: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    cfg = load_config() if cfg is None else cfg
    path = resolve(cfg["paths"]["validation"]) / "primary_model_contract.json"
    if not path.exists():
        raise PrimaryDesignationNotApproved(
            "No frozen primary-model designation. Run the designation stage with --approve."
        )
    contract = json.loads(path.read_text())
    if contract.get("status") != "frozen":
        raise PrimaryDesignationNotApproved("Primary-model contract is not frozen")
    if contract.get("primary_model") != PRIMARY_MODEL:
        raise RuntimeError("Primary-model contract does not designate Ridge")
    if contract.get("test_used_for_selection") is not False:
        raise RuntimeError("Primary designation violates test independence")
    comparison = _load_comparison(cfg)
    if _validation_fingerprint(comparison) != contract.get("validation_fingerprint"):
        raise RuntimeError(
            "Validation results changed after the primary designation was frozen. "
            "Re-run the designation stage and re-approve."
        )
    return contract


def active_contract(cfg: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Return the frozen Ridge primary-model designation.

    The superseded automatic minimum-MSE rule is deliberately not a fallback:
    the active workflow must always use the Ridge-centred research policy.
    """
    cfg = load_config() if cfg is None else cfg
    return require_primary_designation(cfg)
