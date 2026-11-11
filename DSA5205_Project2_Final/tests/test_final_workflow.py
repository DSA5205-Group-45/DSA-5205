"""Final-workflow checks: score-to-position, R4 reconciliation and no active auto-selection."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.research_policy import PrimaryDesignationNotApproved, active_contract

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_score_position_is_a_real_top10_equal_weight_portfolio():
    path = PROJECT_ROOT / "results/oos/score_to_position_example.csv"
    if not path.exists():
        pytest.skip("Score-to-position example not generated")
    table = pd.read_csv(path)
    assert len(table) == 30
    assert int(table.selected.sum()) == 10
    assert np.isclose(table.target_weight.sum(), 1.0)
    assert set(table.loc[table.selected, "target_weight"]) == {0.1}
    assert set(table.loc[~table.selected, "target_weight"]) == {0.0}
    ordered = table.sort_values("rank")
    assert ordered.ticker.tolist() == table.sort_values(["prediction", "ticker"], ascending=[False, True]).ticker.tolist()
    assert ordered.loc[ordered.selected, "rank"].max() == 10
    assert (PROJECT_ROOT / "results/oos/figures/fig_score_to_position.png").exists()
    assert (PROJECT_ROOT / "figures/fig_score_to_position.png").exists()


def test_ridge_r4_reconciliation_is_documented_not_merged():
    path = PROJECT_ROOT / "results/extensions/ridge_estimator_reconciliation.json"
    if not path.exists():
        pytest.skip("R4 reconciliation not generated")
    payload = json.loads(path.read_text())
    assert payload["status"] == "same_economic_rule_different_fitted_model"
    assert payload["main_ridge"]["refit_rows"] == 4260
    assert payload["r4"]["refit_rows"] == 8520
    assert payload["main_ridge"]["candidate_pool_size"] == 30
    assert payload["r4"]["candidate_pool_size"] == 60
    comparison = payload["comparison"]
    assert 0.85 <= comparison["mean_holdings_overlap"] <= 1.0
    assert comparison["return_correlation"] > 0.99
    assert 0.0 < comparison["cagr_difference_r4_minus_main"] < 0.03
    overlap = pd.read_csv(PROJECT_ROOT / "results/extensions/ridge_estimator_holdings_overlap.csv")
    assert len(overlap) == 48
    # R4 must not be counted as a ninth main OOS strategy.
    metrics = pd.read_csv(PROJECT_ROOT / "results/oos/eight_strategy/metrics.csv")
    assert "R4_RidgeDoubleSort30" not in set(metrics.strategy)
    assert len(metrics) == 8


def test_active_contract_has_no_automatic_selection_fallback(tmp_path):
    cfg = {
        "paths": {"validation": str(tmp_path / "validation"), "results": str(tmp_path)},
        "selection": {"primary_model": "ridge"},
    }
    (tmp_path / "validation").mkdir()
    with pytest.raises(PrimaryDesignationNotApproved):
        active_contract(cfg)


def test_cli_has_no_active_select_stage():
    source = (PROJECT_ROOT / "run_pipeline.py").read_text()
    assert '"select"' not in source
    assert "_legacy_select_stage" not in source
    assert "designate" in source


def test_superseded_selection_is_clearly_labelled():
    archive = PROJECT_ROOT / "results/validation/archive/superseded_automatic_selection"
    assert (archive / "README.md").exists()
    manifest = json.loads((archive / "ARCHIVE_MANIFEST.json").read_text())
    assert manifest["status"] == "superseded_auxiliary_artifact"
    assert manifest["active_workflow_uses_this"] is False
    assert manifest["current_primary_model"] == "ridge"
    # The root of results/validation must not contain active-looking old contracts.
    assert not (PROJECT_ROOT / "results/validation/selection_contract.json").exists()
    assert not (PROJECT_ROOT / "results/validation/model_selection_summary.csv").exists()


def test_public_extension_excludes_internal_estimator_sleeve():
    public = pd.read_csv(PROJECT_ROOT / "results/extensions/strategy_summary_public.csv")
    internal = pd.read_csv(PROJECT_ROOT / "results/extensions/strategy_summary.csv")
    assert len(public) == 8
    assert len(internal) == 9
    assert not any("R4" in name for name in public.strategy)
    figures = PROJECT_ROOT / "results/extensions/figures"
    for name in ("extension_cumulative_wealth.png", "extension_annual_returns.png", "extension_drawdowns.png"):
        assert (figures / name).exists(), name
    assert not list(figures.glob("nine_strategy_*.png"))


def test_user_facing_materials_do_not_present_r4_as_a_strategy():
    import nbformat

    tex = (PROJECT_ROOT / "report/Project2_Report.tex").read_text()
    assert "R4" not in tex
    assert "nine-strategy" not in tex.lower()
    notebook = nbformat.read(PROJECT_ROOT / "notebooks/Project2_Final.ipynb", as_version=4)
    source = "\n".join("".join(cell.source) for cell in notebook.cells)
    assert "R4" not in source
    assert "nine-strategy" not in source.lower()
