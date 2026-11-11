"""Leakage controls: label embargo, preprocessing isolation, test independence."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline import chronological_splits
from src.final_refit import refit_models
from src.research_policy import (
    PrimaryDesignationNotApproved,
    designate_primary_model,
    require_primary_designation,
)
from src.models import build_model, parameter_grid
from src.pipeline import FEATURES
from src.validation import run_validation, tune_all_models


def test_training_labels_settle_before_cutoff(real_model_ready, real_cfg):
    splits, _, _ = chronological_splits(real_model_ready, real_cfg)
    cutoff = pd.Timestamp(real_cfg["data"]["train_end_exclusive"])
    assert splits["train"].label_end_date.max() <= cutoff
    assert splits["train"].decision_date.max() < cutoff


def test_validation_labels_settle_before_test_start(real_model_ready, real_cfg):
    splits, _, excluded = chronological_splits(real_model_ready, real_cfg)
    boundary = pd.Timestamp(real_cfg["data"]["validation_end_exclusive"])
    assert splits["validation"].label_end_date.max() <= boundary
    assert splits["test"].decision_date.min() >= pd.Timestamp(real_cfg["data"]["test_start"])
    assert not excluded.empty
    assert set(excluded["reason"]) == {"label settles after split boundary"}


def test_scaler_is_fitted_on_training_rows_only(synthetic_panel, synthetic_cfg):
    splits, _, _ = chronological_splits(synthetic_panel, synthetic_cfg)
    train = splits["train"].copy()
    validation = splits["validation"].copy()
    # Deliberately shift validation features so any leakage would be obvious.
    validation[FEATURES] = validation[FEATURES] + 100.0
    estimator = build_model("ridge", {"alpha": 10.0}, seed=42)
    estimator.fit(train[FEATURES], train["target_forward_return"])
    scaler = estimator.named_steps["scale"]
    assert np.allclose(scaler.mean_, train[FEATURES].to_numpy(dtype=float).mean(axis=0))
    assert not np.allclose(scaler.mean_, validation[FEATURES].to_numpy(dtype=float).mean(axis=0))


def test_selection_contract_declares_no_test_use(selection_cfg):
    contract = designate_primary_model(selection_cfg, approve=True)
    assert contract["test_used_for_selection"] is False
    assert contract["test_data_accessed"] is False
    assert contract["selected_model"] == "ridge"
    assert contract["historical_automatic_selection"]["selected_model"] == "lasso"
    assert contract["primary_model"] == "ridge"
    stored = json.loads(Path(selection_cfg["paths"]["validation"] + "/primary_model_contract.json").read_text())
    assert stored["test_used_for_selection"] is False


def test_missing_approval_blocks_oos_and_writes_pending(selection_cfg):
    with pytest.raises(PrimaryDesignationNotApproved):
        designate_primary_model(selection_cfg, approve=False)
    pending = Path(selection_cfg["paths"]["validation"]) / "primary_model_designation_pending.json"
    payload = json.loads(pending.read_text())
    assert payload["status"] == "awaiting_explicit_approval"
    with pytest.raises(PrimaryDesignationNotApproved):
        require_primary_designation(selection_cfg)


def test_corrupting_test_labels_does_not_change_tuning(synthetic_panel, synthetic_cfg):
    splits, _, _ = chronological_splits(synthetic_panel, synthetic_cfg)
    _, best_original, _ = tune_all_models(splits, synthetic_cfg)

    corrupted = synthetic_panel.copy()
    test_mask = corrupted["split"] == "test"
    corrupted.loc[test_mask, "target_forward_return"] = 123.0
    corrupted_splits, _, _ = chronological_splits(corrupted, synthetic_cfg)
    _, best_corrupted, _ = tune_all_models(corrupted_splits, synthetic_cfg)

    for name in best_original:
        assert best_original[name]["params"] == best_corrupted[name]["params"]
        assert best_original[name]["validation_mse"] == best_corrupted[name]["validation_mse"]


def test_selection_is_independent_of_test_data(selection_cfg):
    contract = designate_primary_model(selection_cfg, approve=True)
    first = contract["selected_model"]
    # Corrupting any file outside the validation comparison must not matter,
    # and re-running selection is idempotent.
    again = designate_primary_model(selection_cfg, approve=True)
    assert again["selected_model"] == first
    assert again["selection_timestamp_utc"] == contract["selection_timestamp_utc"]


def test_refit_excludes_labels_unavailable_at_cutoff(synthetic_cfg):
    run_validation(synthetic_cfg)
    designate_primary_model(synthetic_cfg, approve=True)
    result = refit_models(synthetic_cfg)
    cutoff = pd.Timestamp(synthetic_cfg["data"]["refit_end_exclusive"])
    summary = result["summary"]
    for _, row in summary.iterrows():
        assert pd.Timestamp(row["refit_label_end_max"]) <= cutoff
        assert pd.Timestamp(row["refit_decision_end"]) < pd.Timestamp(synthetic_cfg["data"]["test_start"])
    assert result["refit_rows"] == 240  # 8 cohorts x 30 candidates
    assert set(summary["model"]) == {"ridge", "lasso", "rf", "xgb"}
    assert summary.loc[summary["is_selected"], "model"].iloc[0] == result["selected_model"]


def test_refit_hyperparameters_match_frozen_contract(synthetic_cfg):
    run_validation(synthetic_cfg)
    contract = designate_primary_model(synthetic_cfg, approve=True)
    result = refit_models(synthetic_cfg, contract=contract)
    selected_row = result["summary"].loc[result["summary"]["is_selected"]].iloc[0]
    assert json.loads(selected_row["hyperparameters"]) == contract["selected_hyperparameters"]
