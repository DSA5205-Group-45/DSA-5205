"""Ridge-centred primary-model policy and preservation of the LASSO audit."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.model_selection import select_final_model
from src.research_policy import (
    PrimaryDesignationNotApproved,
    active_contract,
    archive_historical_selection,
    designate_primary_model,
    designation_table,
    require_primary_designation,
)


def _freeze_historical(selection_cfg):
    return select_final_model(selection_cfg, criterion="lowest_validation_mse", approve=True)


def test_designation_requires_approval(selection_cfg):
    _freeze_historical(selection_cfg)
    with pytest.raises(PrimaryDesignationNotApproved):
        designate_primary_model(selection_cfg, approve=False)
    pending = Path(selection_cfg["paths"]["validation"]) / "primary_model_designation_pending.json"
    assert pending.exists()
    payload = json.loads(pending.read_text())
    assert payload["status"] == "awaiting_explicit_approval"
    assert payload["proposed_primary_model"] == "ridge"


def test_designation_freezes_ridge_and_archives_lasso(selection_cfg):
    historical = _freeze_historical(selection_cfg)
    assert historical["selected_model"] == "lasso"
    contract = designate_primary_model(selection_cfg, approve=True)
    assert contract["primary_model"] == "ridge"
    assert contract["selected_model"] == "ridge"
    assert contract["selected_hyperparameters"] == {"alpha": 1000.0}
    assert contract["hyperparameter_rule"] == "lowest_validation_mse_within_ridge"
    assert contract["test_used_for_selection"] is False
    assert contract["test_metrics_used_in_designation"] is False
    assert contract["clean_preregistration"] is False
    assert contract["designation_informed_by_prior_oos_review"] is True
    assert contract["historical_automatic_selection"]["selected_model"] == "lasso"
    archive = contract["historical_automatic_selection"]["archived_manifest"]
    assert archive["files"]
    archived = Path(selection_cfg["paths"]["validation"]) / "archive" / "superseded_automatic_selection" / "selection_contract.json"
    assert archived.exists()
    assert json.loads(archived.read_text())["selected_model"] == "lasso"


def test_historical_contract_is_not_overwritten(selection_cfg):
    historical = _freeze_historical(selection_cfg)
    path = Path(selection_cfg["paths"]["validation"]) / "selection_contract.json"
    before = json.loads(path.read_text())
    designate_primary_model(selection_cfg, approve=True)
    after = json.loads(path.read_text())
    assert before["selected_model"] == "lasso"
    assert after["selected_model"] == "lasso"
    assert historical["selected_model"] == "lasso"


def test_designation_table_roles(selection_cfg):
    _freeze_historical(selection_cfg)
    designate_primary_model(selection_cfg, approve=True)
    table = designation_table(selection_cfg).set_index("model")
    assert table.loc["ridge", "research_role"] == "Primary research model"
    assert bool(table.loc["ridge", "is_primary"]) is True
    for model in ("lasso", "rf", "xgb"):
        assert table.loc[model, "research_role"] == "Alternative model"
        assert bool(table.loc[model, "is_primary"]) is False


def test_require_primary_designation_detects_changed_validation(selection_cfg):
    _freeze_historical(selection_cfg)
    designate_primary_model(selection_cfg, approve=True)
    require_primary_designation(selection_cfg)
    path = Path(selection_cfg["paths"]["validation"]) / "validation_model_comparison.csv"
    frame = pd.read_csv(path)
    frame.loc[frame.model == "ridge", "validation_mse"] = 0.99
    frame.to_csv(path, index=False)
    with pytest.raises(RuntimeError, match="Validation results changed"):
        require_primary_designation(selection_cfg)


def test_active_contract_requires_primary_designation(selection_cfg):
    _freeze_historical(selection_cfg)
    with pytest.raises(PrimaryDesignationNotApproved):
        active_contract(selection_cfg)
    designate_primary_model(selection_cfg, approve=True)
    assert active_contract(selection_cfg)["selected_model"] == "ridge"


def test_archive_manifest_records_reason(selection_cfg):
    _freeze_historical(selection_cfg)
    manifest = archive_historical_selection(selection_cfg)
    assert manifest["files"]
    assert "automatic lowest-validation-MSE" in manifest["reason"]
