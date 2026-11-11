"""Final-refit contract: cutoff, fixed hyperparameters, artifacts and metadata."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.final_refit import load_refit_models, load_validation_best_params, refit_models
from src.research_policy import PrimaryDesignationNotApproved, designate_primary_model
from src.models import MODEL_NAMES
from src.pipeline import FEATURES
from src.validation import run_validation


@pytest.fixture
def fitted_environment(synthetic_cfg):
    run_validation(synthetic_cfg)
    contract = designate_primary_model(synthetic_cfg, approve=True)
    result = refit_models(synthetic_cfg, contract=contract)
    return synthetic_cfg, contract, result


def test_refit_requires_a_frozen_contract(synthetic_cfg):
    run_validation(synthetic_cfg)
    with pytest.raises(PrimaryDesignationNotApproved):
        refit_models(synthetic_cfg)


def test_refit_summary_has_all_four_models(fitted_environment):
    cfg, contract, result = fitted_environment
    summary = result["summary"]
    assert set(summary["model"]) == set(MODEL_NAMES)
    assert summary["is_selected"].sum() == 1
    assert summary.loc[summary["is_selected"], "model"].iloc[0] == "ridge"
    assert (summary["n_training_observations"] == 240).all()


def test_refit_respects_the_label_cutoff(fitted_environment):
    cfg, _, result = fitted_environment
    summary = result["summary"]
    cutoff = pd.Timestamp(cfg["data"]["refit_end_exclusive"])
    test_start = pd.Timestamp(cfg["data"]["test_start"])
    assert (pd.to_datetime(summary["refit_label_end_max"]) <= cutoff).all()
    assert (pd.to_datetime(summary["refit_decision_end"]) < test_start).all()


def test_refit_hyperparameters_are_frozen(fitted_environment):
    cfg, contract, result = fitted_environment
    best = load_validation_best_params(cfg)
    summary = result["summary"].set_index("model")
    for name in MODEL_NAMES:
        assert json.loads(summary.loc[name, "hyperparameters"]) == best[name]["params"]
    assert json.loads(summary.loc[contract["selected_model"], "hyperparameters"]) == contract["selected_hyperparameters"]


def test_metadata_records_required_fields(fitted_environment):
    cfg, contract, result = fitted_environment
    metadata = result["metadata"]
    for field in (
        "selected_model",
        "selection_criterion",
        "feature_names",
        "random_seed",
        "package_versions",
        "refit_period",
        "n_training_observations",
        "models",
    ):
        assert field in metadata, field
    assert metadata["feature_names"] == FEATURES
    assert metadata["random_seed"] == cfg["project"]["random_seed"]
    assert metadata["selected_model"] == contract["selected_model"]
    assert metadata["test_data_accessed"] is False
    assert metadata["refit_period"]["label_end_max"] <= cfg["data"]["refit_end_exclusive"]
    assert metadata["package_versions"]["python"]


def test_serialized_estimators_load_and_predict(fitted_environment):
    cfg, _, _ = fitted_environment
    estimators = load_refit_models(cfg)
    assert set(estimators) == set(MODEL_NAMES)
    X = pd.DataFrame(np.zeros((3, len(FEATURES))), columns=FEATURES)
    for name, estimator in estimators.items():
        prediction = estimator.predict(X)
        assert prediction.shape == (3,)
        assert np.isfinite(prediction).all()
    for name in MODEL_NAMES:
        assert (Path(cfg["paths"]["models"]) / "refit" / f"{name}.joblib").exists()


def test_refit_summary_file_is_written(fitted_environment):
    cfg, _, _ = fitted_environment
    path = Path(cfg["paths"]["validation"]) / "final_refit_summary.csv"
    assert path.exists()
    frame = pd.read_csv(path)
    assert len(frame) == 4
    metadata = Path(cfg["paths"]["validation"]) / "final_model_metadata.json"
    assert metadata.exists()
    assert json.loads(metadata.read_text())["stage"] == "final_refit"
