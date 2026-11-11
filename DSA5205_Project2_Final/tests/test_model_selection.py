"""Model-selection rules, contract content and guard rails."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from src.model_selection import (
    SelectionNotApproved,
    load_validation_comparison,
    require_selection,
    select_final_model,
)


def _write_comparison(cfg, frame: pd.DataFrame) -> None:
    frame.to_csv(Path(cfg["paths"]["validation"]) / "validation_model_comparison.csv", index=False)


def test_default_rule_selects_lowest_validation_mse(selection_cfg):
    contract = select_final_model(selection_cfg, criterion="lowest_validation_mse", approve=True)
    assert contract["selected_model"] == "lasso"
    assert contract["selection_criterion"] == "lowest_validation_mse"
    assert contract["validation_mse_winner"] == "lasso"


def test_alternative_rule_selects_highest_validation_sharpe(selection_cfg):
    frame = load_validation_comparison(selection_cfg)
    frame.loc[frame.model == "rf", "validation_mse"] = 0.01      # RF wins MSE
    frame.loc[frame.model == "lasso", "validation_sharpe"] = 0.5  # but loses Sharpe
    frame.loc[frame.model == "xgb", "validation_sharpe"] = 2.0    # XGB wins Sharpe
    _write_comparison(selection_cfg, frame)
    contract = select_final_model(selection_cfg, criterion="highest_validation_sharpe", approve=True)
    assert contract["selected_model"] == "xgb"
    assert contract["validation_mse_winner"] == "rf"
    assert contract["validation_sharpe_winner"] == "xgb"
    assert contract["selection_criterion"] == "highest_validation_sharpe"


def test_contract_contains_all_required_fields(selection_cfg, real_cfg):
    contract = select_final_model(selection_cfg, criterion="lowest_validation_mse", approve=True)
    required = [
        "primary_model",
        "selection_criterion",
        "selected_model",
        "selected_hyperparameters",
        "validation_period",
        "selection_timestamp_utc",
        "run_id",
        "test_used_for_selection",
    ]
    for field in required:
        assert field in contract, field
    assert contract["primary_model"] == real_cfg["selection"]["primary_model"] == "ridge"
    assert contract["test_used_for_selection"] is False
    assert contract["selected_hyperparameters"] == {"alpha": 0.001}
    assert contract["status"] == "frozen"


def test_unapproved_selection_writes_pending_and_raises(selection_cfg):
    with pytest.raises(SelectionNotApproved):
        select_final_model(selection_cfg, criterion="lowest_validation_mse", approve=False)
    pending = Path(selection_cfg["paths"]["validation"]) / "model_selection_pending.json"
    payload = json.loads(pending.read_text())
    assert payload["status"] == "awaiting_explicit_approval"
    assert payload["proposed_model"] == "lasso"


def test_require_selection_blocks_missing_contract(selection_cfg):
    with pytest.raises(SelectionNotApproved):
        require_selection(selection_cfg)


def test_require_selection_detects_changed_validation(selection_cfg):
    select_final_model(selection_cfg, criterion="lowest_validation_mse", approve=True)
    frame = load_validation_comparison(selection_cfg)
    frame["validation_mse"] = frame["validation_mse"] * 1.5
    _write_comparison(selection_cfg, frame)
    with pytest.raises(RuntimeError, match="Validation results changed"):
        require_selection(selection_cfg)


def test_selection_summary_ranks(selection_cfg):
    select_final_model(selection_cfg, criterion="lowest_validation_mse", approve=True)
    summary = pd.read_csv(Path(selection_cfg["paths"]["validation"]) / "model_selection_summary.csv")
    assert summary["selected"].sum() == 1
    assert summary.loc[summary.selected, "model"].iloc[0] == "lasso"
    assert summary.loc[summary.model == "lasso", "mse_rank"].iloc[0] == 1
    assert summary.sort_values("selection_rank")["selected"].tolist() == [True, False, False, False]


def test_unknown_criterion_is_rejected(selection_cfg):
    with pytest.raises(ValueError):
        select_final_model(selection_cfg, criterion="highest_test_sharpe", approve=True)
