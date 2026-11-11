"""Four-model contract: fit/predict, identical inputs, deterministic grids."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline import chronological_splits
from src.models import (
    GRIDS,
    MODEL_NAMES,
    build_model,
    convergence_info,
    parameter_grid,
    total_grid_size,
)
from src.pipeline import FEATURES
from src.validation import run_validation, tune_all_models


def _splits(panel, cfg):
    splits, _, _ = chronological_splits(panel, cfg)
    return splits


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_fits_and_predicts(name, synthetic_panel, synthetic_cfg):
    splits = _splits(synthetic_panel, synthetic_cfg)
    train = splits["train"]
    validation = splits["validation"]
    params = parameter_grid(name, synthetic_cfg["models"][name])[0]
    estimator = build_model(name, params, seed=synthetic_cfg["project"]["random_seed"])
    estimator.fit(train[FEATURES], train["target_forward_return"])
    prediction = estimator.predict(validation[FEATURES])
    assert prediction.shape == (len(validation),)
    assert np.isfinite(prediction).all()


def test_all_models_share_features_and_target(synthetic_panel, synthetic_cfg):
    splits = _splits(synthetic_panel, synthetic_cfg)
    train = splits["train"]
    seen = {}
    for name in MODEL_NAMES:
        params = parameter_grid(name, synthetic_cfg["models"][name])[0]
        estimator = build_model(name, params, seed=42)
        estimator.fit(train[FEATURES], train["target_forward_return"])
        seen[name] = list(getattr(estimator, "feature_names_in_", FEATURES))
    assert all(names == FEATURES for names in seen.values())
    assert len({len(names) for names in seen.values()}) == 1


def test_hyperparameter_grids_are_deterministic_and_small(synthetic_cfg):
    expected = {"ridge": 6, "lasso": 5, "rf": 4, "xgb": 8}
    for name in MODEL_NAMES:
        first = parameter_grid(name, synthetic_cfg["models"][name])
        second = parameter_grid(name, synthetic_cfg["models"][name])
        assert first == second
        assert len(first) == expected[name]
        assert len(first) == len(GRIDS[name])
    assert total_grid_size(synthetic_cfg) == 23


def test_rf_and_xgb_grids_include_required_dimensions(synthetic_cfg):
    rf = parameter_grid("rf", synthetic_cfg["models"]["rf"])
    assert {"max_depth", "min_samples_leaf"} <= set(rf[0])
    xgb = parameter_grid("xgb", synthetic_cfg["models"]["xgb"])
    assert {"max_depth", "learning_rate", "n_estimators"} <= set(xgb[0])


def test_best_configuration_minimises_validation_mse(synthetic_panel, synthetic_cfg):
    splits = _splits(synthetic_panel, synthetic_cfg)
    search, best, _ = tune_all_models(splits, synthetic_cfg, criterion="lowest_validation_mse")
    for name in MODEL_NAMES:
        block = search.loc[search.model == name]
        selected = block.loc[block.selected]
        assert len(selected) == 1
        assert selected.iloc[0]["validation_mse"] == block["validation_mse"].min()
        assert best[name]["params"] == json.loads(selected.iloc[0]["params"])


def test_lasso_convergence_information_is_recorded(synthetic_panel, synthetic_cfg):
    splits = _splits(synthetic_panel, synthetic_cfg)
    train = splits["train"]
    estimator = build_model("lasso", {"alpha": 0.001}, seed=42)
    estimator.fit(train[FEATURES], train["target_forward_return"])
    info = convergence_info("lasso", estimator)
    assert np.isfinite(info["n_iter"])
    assert info["converged"] is True
    assert info["n_nonzero_features"] >= 0


def test_validation_prediction_shapes_and_identifiers(synthetic_cfg):
    run_validation(synthetic_cfg)
    predictions = pd.read_csv(f"{synthetic_cfg['paths']['validation']}/validation_predictions.csv")
    assert set(predictions["model"]) == set(MODEL_NAMES)
    counts = predictions.groupby("model").size()
    assert counts.nunique() == 1
    keys = {}
    for name, block in predictions.groupby("model"):
        keys[name] = set(zip(block["decision_date"], block["ticker"]))
        assert block.groupby("decision_date")["pred_rank"].apply(lambda s: sorted(s)).apply(
            lambda x: x == list(range(1, 31))
        ).all()
    reference = keys["ridge"]
    assert all(value == reference for value in keys.values())


def test_seed_makes_tree_models_repeatable(synthetic_panel, synthetic_cfg):
    splits = _splits(synthetic_panel, synthetic_cfg)
    train = splits["train"]
    validation = splits["validation"]
    for name in ("rf", "xgb"):
        params = parameter_grid(name, synthetic_cfg["models"][name])[0]
        first = build_model(name, params, seed=42).fit(train[FEATURES], train["target_forward_return"])
        second = build_model(name, params, seed=42).fit(train[FEATURES], train["target_forward_return"])
        assert np.allclose(first.predict(validation[FEATURES]), second.predict(validation[FEATURES]))
