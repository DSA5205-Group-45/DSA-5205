"""Reproducibility: seeds, deterministic runs, required artefacts, notebook."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline import chronological_splits
from src.research_policy import require_primary_designation
from src.models import build_model, parameter_grid
from src.pipeline import FEATURES
from src.validation import run_validation

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_VALIDATION_FILES = [
    "primary_model_contract.json",
    "primary_model_designation.csv",
    "validation_prediction_metrics.csv",
    "validation_portfolio_metrics.csv",
    "validation_hyperparameter_search.csv",
    "validation_model_comparison.csv",
    "validation_monthly_returns.csv",
    "validation_predictions.csv",
    "final_refit_summary.csv",
    "final_model_metadata.json",
]

REQUIRED_OOS_FILES = [
    "oos_prediction_metrics.csv",
    "oos_portfolio_metrics.csv",
    "oos_model_comparison.csv",
    "oos_monthly_returns.csv",
    "oos_predictions.csv",
    "validation_to_oos_degradation.csv",
    "oos_run_metadata.json",
]

REQUIRED_SCORE_FILES = [
    "score_to_position_example.csv",
    "score_to_position_metadata.json",
]

REQUIRED_RECONCILIATION_FILES = [
    "ridge_estimator_reconciliation.json",
    "ridge_estimator_holdings_overlap.csv",
]

REQUIRED_EIGHT_STRATEGY_FILES = [
    "monthly_returns.csv",
    "metrics.csv",
    "annual_returns.csv",
    "comparison.csv",
    "figure_index.csv",
    "alignment_checks.json",
]

EXPECTED_VALIDATION_FIGURES = 15
EXPECTED_OOS_FIGURES = 12


def test_fixed_seed_repeats_tree_predictions(synthetic_panel, synthetic_cfg):
    splits, _, _ = chronological_splits(synthetic_panel, synthetic_cfg)
    train, validation = splits["train"], splits["validation"]
    for name in ("rf", "xgb"):
        params = parameter_grid(name, synthetic_cfg["models"][name])[0]
        first = build_model(name, params, 42).fit(train[FEATURES], train["target_forward_return"])
        second = build_model(name, params, 42).fit(train[FEATURES], train["target_forward_return"])
        # RandomForest aggregates trees in parallel, so repeated fits can differ
        # by float rounding (~1e-16); XGBoost with a fixed seed is bit-identical.
        assert np.allclose(
            first.predict(validation[FEATURES]),
            second.predict(validation[FEATURES]),
            rtol=0.0,
            atol=1e-12,
        )


def test_validation_is_deterministic(synthetic_cfg):
    first = run_validation(synthetic_cfg)
    first_comparison = first["comparison"].copy()
    second = run_validation(synthetic_cfg)
    pd.testing.assert_frame_equal(first_comparison, second["comparison"])


def test_required_validation_outputs_exist():
    validation = PROJECT_ROOT / "results" / "validation"
    if not validation.exists():
        pytest.skip("Validation results not generated in this checkout")
    for name in REQUIRED_VALIDATION_FILES:
        assert (validation / name).exists(), name
    archive = validation / "archive" / "superseded_automatic_selection"
    assert (archive / "selection_contract.json").exists()
    assert (archive / "README.md").exists()
    figures = sorted((validation / "figures").glob("*.png"))
    assert len(figures) == EXPECTED_VALIDATION_FIGURES
    assert figures[0].name.startswith("01_")
    assert figures[-1].name.startswith("15_")


def test_required_oos_outputs_exist():
    oos = PROJECT_ROOT / "results" / "oos"
    if not oos.exists():
        pytest.skip("OOS results not generated in this checkout")
    for name in REQUIRED_OOS_FILES:
        assert (oos / name).exists(), name
    figures = sorted((oos / "figures").glob("*.png"))
    assert len(figures) == EXPECTED_OOS_FIGURES


def test_required_extension_outputs_exist():
    extension = PROJECT_ROOT / "results" / "extensions"
    if not extension.exists():
        pytest.skip("Extension results not generated in this checkout")
    for name in (
        "strategy_summary.csv",
        "period_returns.csv",
        "annual_returns.csv",
        "validation_grid.csv",
        "fit_info.json",
        "prediction_metrics.json",
    ):
        assert (extension / name).exists(), name
    assert len(list((extension / "figures").glob("*.png"))) >= 5


def test_real_selection_contract_is_test_independent():
    contract_path = PROJECT_ROOT / "results" / "validation" / "primary_model_contract.json"
    if not contract_path.exists():
        pytest.skip("Primary-model contract not generated in this checkout")
    contract = require_primary_designation()
    assert contract["test_used_for_selection"] is False
    assert contract["test_data_accessed"] is False
    assert contract["status"] == "frozen"


def test_score_position_outputs_exist():
    oos = PROJECT_ROOT / "results" / "oos"
    if not oos.exists():
        pytest.skip("OOS results not generated in this checkout")
    for name in REQUIRED_SCORE_FILES:
        assert (oos / name).exists(), name
    assert (oos / "figures" / "fig_score_to_position.png").exists()


def test_ridge_r4_reconciliation_outputs_exist():
    extension = PROJECT_ROOT / "results" / "extensions"
    if not extension.exists():
        pytest.skip("Extension results not generated in this checkout")
    for name in REQUIRED_RECONCILIATION_FILES:
        assert (extension / name).exists(), name
    payload = json.loads((extension / "ridge_estimator_reconciliation.json").read_text())
    assert payload["status"] == "same_economic_rule_different_fitted_model"
    assert payload["main_ridge"]["refit_rows"] == 4260
    assert payload["r4"]["refit_rows"] == 8520
    assert 0.5 < payload["comparison"]["mean_holdings_overlap"] <= 1.0


def test_eight_strategy_outputs_exist():
    eight = PROJECT_ROOT / "results" / "oos" / "eight_strategy"
    if not eight.exists():
        pytest.skip("Eight-strategy results not generated in this checkout")
    for name in REQUIRED_EIGHT_STRATEGY_FILES:
        assert (eight / name).exists(), name
    figures = PROJECT_ROOT / "results" / "oos" / "figures"
    for name in [f"fig{i}_" for i in range(1, 7)]:
        matches = list(figures.glob(f"{name}*.png"))
        assert len(matches) == 1, name


def test_notebook_structure():
    notebook_path = PROJECT_ROOT / "notebooks" / "Project2_Final.ipynb"
    assert notebook_path.exists()
    import nbformat

    notebook = nbformat.read(notebook_path, as_version=4)
    text = "\n".join("".join(cell.source) for cell in notebook.cells)
    required_sections = [
        "Research Question and Economic Motivation",
        "Markets, Assets, Trading Calendar and Benchmarks",
        "Data Sources, Cleaning and Universe Construction",
        "Labels, Features and Timing Alignment",
        "Feature Correlation and Ridge Model Justification",
        "Training, Validation and Hyperparameter Tuning",
        "Four-Model Validation Comparison",
        "Primary Ridge Strategy and Trading Rules",
        "Final OOS Evaluation: Four ML Models",
        "Final OOS Evaluation: Ridge vs B1, B3, SPY and QQQ",
        "Interpretation and Limitations",
        "Exploratory Ridge Strategy Extensions",
        "Conclusion",
        "Reproducibility and AI-Use Declaration",
    ]
    for section in required_sections:
        assert section.lower() in text.lower(), f"missing notebook section: {section}"
    assert "/Users/hc" not in text, "notebook must use portable relative paths"


@pytest.mark.skipif(
    os.environ.get("RUN_NOTEBOOK_TEST") != "1",
    reason="set RUN_NOTEBOOK_TEST=1 to execute the full notebook",
)
def test_notebook_executes_from_clean_kernel():
    import nbformat
    from nbclient import NotebookClient

    notebook_path = PROJECT_ROOT / "notebooks" / "Project2_Final.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=1200,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
        allow_errors=False,
    )
    client.execute()
    errors = [
        output
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.output_type == "error"
    ]
    assert not errors, errors
