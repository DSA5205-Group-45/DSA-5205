# Submission Manifest

**Project:** DSA5205 Project 2 — Ridge-Based Cross-Sectional Equity Strategy
**Final directory:** `DSA5205_Project2_Final/`
**Submission ZIP:** `DSA5205_Project2_Final_Submission.zip`
**Date:** 2026-11-11

This manifest uses the actual paths in the final project. It separates what must be submitted from what should remain local, what may be deleted, what was archived and what is excluded because of size.

---

## 1. Files that must be submitted (Category A)

All paths are relative to `DSA5205_Project2_Final/`.

### Top-level documents and entry points

| Path | Purpose |
|---|---|
| `README.md` | Research summary, installation, exact reproduction commands, limitations |
| `requirements.txt` | Python dependencies |
| `CHANGELOG.md` | Revision history |
| `INTEGRATION_AUDIT.md` | Provenance, technical verification, test record |
| `COMPLIANCE_CHECKLIST.md` | §4.1–4.9 and rubric mapping |
| `SUBMISSION_MANIFEST.md` | This file |
| `run_pipeline.py` | End-to-end pipeline CLI |
| `run_tests.py` | Test-suite wrapper |

### Configuration

| Path | Purpose |
|---|---|
| `config/experiment_config.yaml` | Dates, embargo, grids, portfolio rules, output paths |
| `config/pipeline.json` | Original pipeline parameters |
| `config/preserved_sources.json` | Hash locks for preserved code |

### Source code

| Path | Purpose |
|---|---|
| `src/pipeline.py` | Preserved data/feature/screening/label pipeline |
| `src/backtest.py` | Original trading rules plus comparison helpers |
| `src/qa.py`, `src/reporting.py`, `src/visualize.py` | Preserved QA/reporting helpers |
| `src/data_pipeline.py` | Loading, splits, embargo, derived panel |
| `src/features.py` | Feature registry and preprocessing contract |
| `src/models.py` | Four-model factories, grids, convergence |
| `src/validation.py` | Tuning and validation predictions |
| `src/metrics.py` | Prediction and portfolio metrics |
| `src/model_selection.py` | Legacy auxiliary module (not in active workflow) |
| `src/research_policy.py` | Ridge primary-model designation |
| `src/diagnostics.py` | Correlation and coefficient-stability diagnostics |
| `src/final_refit.py` | 2010–2021 refit and serialization |
| `src/oos_evaluation.py` | Four-model OOS evaluation |
| `src/oos_comparison.py` | Eight-strategy reconciliation and Figures 1–6 |
| `src/score_position.py` | Figure 1 score-to-position example |
| `src/reconciliation.py` | Internal Ridge-estimator reconciliation (reproducibility only) |
| `src/ridge_extension.py` | Nine-strategy extension driver |
| `src/visualization.py` | Validation and OOS dashboards |
| `src/utils.py` | Hashing, timestamps, environment metadata |
| `strategies.py` | Preserved exploratory extension (nine internal sleeves) |

### Notebook

| Path | Purpose |
|---|---|
| `notebooks/Project2_Final.ipynb` | Executed 14-section notebook (40 cells, 0 errors, 14 embedded figures) |

### Reproducibility inputs

| Path | Purpose |
|---|---|
| `data/processed_ready/monthly/model_ready.parquet` | Monthly model-ready panel (5,760 rows, 192 dates) |
| `data/processed_ready/monthly/screened_with_label_status.parquet` | Screening audit with label status |
| `data/processed_ready/monthly/feature_columns.json` | Feature contract |
| `data/processed_ready/monthly/model_contract.json` | Original metadata (contains a known stale split field; see audit) |
| `data/processed_ready/quarterly/*`, `data/processed_ready/semiannual/*` | Optional frequency panels retained for completeness |
| `data/derived/extension_top300_labeled.parquet` | Compact 300-name candidate panel for the extension |
| `data/derived/extension_panel_meta.json` | Provenance of the derived panel |
| `data/derived/extension_screening_audit.csv` | Monthly eligibility audit |

### Results

| Path | Purpose |
|---|---|
| `results/validation/primary_model_contract.json` | Ridge designation, rationale, disclosure |
| `results/validation/primary_model_designation.csv` | Research roles of the four models |
| `results/validation/validation_prediction_metrics.csv` | MSE, R², Rank IC, direction |
| `results/validation/validation_portfolio_metrics.csv` | Portfolio metrics |
| `results/validation/validation_hyperparameter_search.csv` | 23 configurations |
| `results/validation/validation_model_comparison.csv` | Four-model comparison |
| `results/validation/validation_monthly_returns.csv` | Monthly returns |
| `results/validation/validation_predictions.csv` | All validation predictions |
| `results/validation/final_refit_summary.csv` | Refit record |
| `results/validation/final_model_metadata.json` | Refit metadata and hashes |
| `results/validation/split_summary.csv` | Split coverage |
| `results/validation/label_embargo_excluded.csv` | Excluded cohorts |
| `results/validation/data_quality.json` | Data-quality checks |
| `results/validation/figures/01…15_*.png` | 15 validation figures |
| `results/validation/archive/superseded_automatic_selection/selection_contract.json` | Historical LASSO audit (read-only) |
| `results/validation/archive/superseded_automatic_selection/model_selection_summary.csv` | Historical LASSO summary |
| `results/validation/archive/superseded_automatic_selection/README.md` | “Superseded artifact” label |
| `results/validation/archive/superseded_automatic_selection/ARCHIVE_MANIFEST.json` | Archive metadata |
| `results/oos/oos_prediction_metrics.csv` | OOS prediction metrics |
| `results/oos/oos_portfolio_metrics.csv` | OOS portfolio metrics |
| `results/oos/oos_model_comparison.csv` | OOS model comparison |
| `results/oos/oos_predictions.csv` | All OOS predictions |
| `results/oos/oos_monthly_returns.csv` | OOS monthly returns |
| `results/oos/validation_to_oos_degradation.csv` | Degradation analysis |
| `results/oos/oos_run_metadata.json`, `final_selection_audit.json` | OOS audit |
| `results/oos/score_to_position_example.csv` | Real 2025-06-30 score/weight example |
| `results/oos/score_to_position_metadata.json` | Example metadata |
| `results/oos/eight_strategy/monthly_returns.csv` | Eight-strategy panel |
| `results/oos/eight_strategy/metrics.csv` | Eight-strategy metrics |
| `results/oos/eight_strategy/annual_returns.csv` | Annual returns |
| `results/oos/eight_strategy/comparison.csv` | Compact comparison |
| `results/oos/eight_strategy/figure_index.csv` | Figure-to-data map |
| `results/oos/eight_strategy/alignment_checks.json` | Cross-engine checks |
| `results/oos/figures/fig1…fig6_*.png` | Six OOS exhibits |
| `results/oos/figures/fig_score_to_position.png` | Figure 1 source |
| `results/diagnostics/feature_correlation_matrix.csv` | Training-period correlations |
| `results/diagnostics/feature_correlation_pairs.csv` | Strong pairs |
| `results/diagnostics/rolling_coefficients.csv` | Rolling coefficients |
| `results/diagnostics/coefficient_changes_by_window.csv` | Change series |
| `results/diagnostics/coefficient_stability_summary.csv` | Ridge/LASSO stability |
| `results/diagnostics/report_correlation_heatmap.png` | Appendix A1 |
| `results/diagnostics/report_stability_comparison.png` | Appendix A2 |
| `results/extensions/strategy_summary_public.csv` | Eight-strategy public summary (excludes internal estimator sleeve) |
| `results/extensions/period_returns_public.csv` | Public extension monthly returns |
| `results/extensions/annual_returns_public.csv` | Public extension annual returns |
| `results/extensions/strategy_summary.csv` | Internal nine-row audit record |
| `results/extensions/period_returns.csv` | Extension monthly returns |
| `results/extensions/annual_returns.csv` | Extension annual returns |
| `results/extensions/holdings.csv`, `trades.csv` | Holdings and trade log |
| `results/extensions/validation_grid.csv`, `fit_info.json` | Extension Ridge record |
| `results/extensions/prediction_metrics.json`, `ridge_rank_ic.csv` | Extension prediction diagnostics |
| `results/extensions/ridge_test60_predictions.csv` | Extension predictions |
| `results/extensions/screening_audit.csv` | Extension eligibility audit |
| `results/models/refit/ridge.joblib` | Final Ridge estimator |
| `results/models/refit/lasso.joblib` | Comparison estimator |
| `results/models/refit/rf.joblib` | Comparison estimator |
| `results/models/refit/xgb.joblib` | Comparison estimator |

### Report and curated figures

| Path | Purpose |
|---|---|
| `report/Project2_Report.pdf` | 23-page academic report |
| `report/Project2_Report.tex` | Editable LaTeX source |
| `report/build_report.py` | Rebuilds report from `results/` |
| `report/generated/table_*.tex` | Generated LaTeX tables |
| `figures/fig_score_to_position.png` | Figure 1 |
| `figures/fig_four_model.png` | Figure 2 |
| `figures/fig_ridge_benchmarks.png` | Figure 3 |
| `figures/fig_eight_summary.png` | Figure 4 |
| `figures/fig_drawdowns.png` | Figure 5 |
| `figures/fig_annual.png` | Figure 6 |
| `figures/fig_risk_adjusted.png` | Figure 7 |
| `figures/fig_correlation.png` | Appendix A1 |
| `figures/fig_stability.png` | Appendix A2 |

### Tests

All 15 files under `tests/`:

`conftest.py`, `extension_fixture.py`, `fixtures/extension_golden.json`, `test_data_integrity.py`, `test_data_leakage.py`, `test_four_models.py`, `test_model_selection.py`, `test_research_policy.py`, `test_backtest.py`, `test_final_refit.py`, `test_ridge_extension.py`, `test_eight_strategy.py`, `test_final_workflow.py`, `test_reproducibility.py`, `test_report.py`.

---

## 2. Files that should remain local (Category B)

| Path | Reason |
|---|---|
| `<original_project>/data/raw/market_data.parquet` | ≈57 MB raw Yahoo Finance snapshot; excluded from the submission, used to rebuild the derived panels |
| `<original_project>/data/processed_with_nan/full_feature_panel.parquet` | ≈331 MB intermediate feature panel; excluded |
| `<original_project>/data/raw/cache/*.parquet` | Per-ticker download cache (≈505 files); excluded |
| `.venv/` | Local virtual environment |
| `report/Project2_Report.aux`, `.log`, `.out`, `.toc` | LaTeX build products; regenerated by `python report/build_report.py` |
| `.pytest_cache/`, `__pycache__/` | Python caches; regenerated |

---

## 3. Files that may be deleted locally (Category D)

| Path | Reason |
|---|---|
| `report/Project2_Report.aux`, `.log`, `.out`, `.toc` | Build products |
| `.pytest_cache/` | Pytest cache |
| `**/__pycache__/` | Python bytecode |
| `**/.DS_Store` | macOS metadata |
| `notebooks/.ipynb_checkpoints/` | Notebook checkpoints (none present) |
| Duplicate ZIPs in the parent directory (`DSA5205_Project2_Integrated.zip`, older `DSA5205_Project2_Final_Submission.zip` if regenerated) | Superseded packages |

Do **not** delete `results/`, `figures/`, `report/Project2_Report.pdf`, the fitted models or the `data/processed_ready` and `data/derived` panels: they are required to inspect and verify the reported results.

---

## 4. Files moved to local archive (Category C)

| Path | Reason |
|---|---|
| `../_archive/DSA5205_Project2_Integrated_pre_final/` | Full pre-final project copy, preserved outside the submission directory |
| `<original_project>/notebooks/Project2_Four_Model_Complete.ipynb` | Superseded 49-cell notebook, retained in the original project |
| `<original_project>/outputs/` and `outputs.zip` | Legacy single-model `run_model` outputs, not used by the final pipeline |
| `<original_project>/results.zip`, `results/research_diagnostics.zip` | Duplicate archives of the same results |

These are not part of the submission ZIP.

Internal reproducibility artifacts retained in the submission: `results/extensions/ridge_estimator_reconciliation.json`, `results/extensions/ridge_estimator_holdings_overlap.csv`, and `results/extensions/figures/fig_internal_estimator_reconciliation.png`. They document the estimator difference and are not used in the report or the public strategy tables.

---

## 5. Files intentionally excluded because of size (Category B/E)

| Path | Size | Why excluded | How to regenerate |
|---|---:|---|---|
| `data/raw/market_data.parquet` | ≈57 MB | Raw vendor snapshot | Re-download with `src.pipeline.download` / original pipeline, or copy from the team archive |
| `data/processed_with_nan/full_feature_panel.parquet` | ≈331 MB | Intermediate feature panel | `python -m src.data_pipeline --rebuild-panel --frequency monthly` |
| `data/raw/cache/` | ≈50 MB | Per-ticker cache | Re-created by the download pipeline |

The included `data/processed_ready` and `data/derived` panels are sufficient to run every model, backtest, test and figure in this package.

---

## 6. Required execution commands

```bash
cd DSA5205_Project2_Final
python -m pip install -r requirements.txt

python run_pipeline.py --stage validation
python run_pipeline.py --stage designate --approve
python run_pipeline.py --stage refit
python run_pipeline.py --stage oos
python run_pipeline.py --stage score
python run_pipeline.py --stage extension
python run_pipeline.py --stage eight

python run_tests.py --notebook
python report/build_report.py
```

The primary result is complete after `--stage oos`; `--stage extension` and `--stage eight` add the exploratory and benchmark comparisons. The report requires the results from all stages.

---

## 7. Verification status

| Check | Result |
|---|---|
| Python test suite (`RUN_NOTEBOOK_TEST=1 pytest -q`) | **115 passed, 0 failed** |
| Notebook clean-kernel execution | 40 cells, 0 errors, 14 embedded figures |
| Report page count | 23 pages (15–25 required) |
| Report undefined references | 0 |
| Report overfull boxes | 0 |
| Report figures/tables referenced | all labels referenced |
| Data rows and dates | 5,760 candidate rows, 192 monthly dates, 30 names/date |
| Training/validation/test cohorts | 118 / 22 / 48 |
| Ridge refit cutoff | labels settle by 2021-12-01; no test rows |
| Cost verification | first-period turnover 1.0, cost 0.002; drift arithmetic tested |
| Benchmark cross-engine equality | SPY/QQQ max difference 0.0 |
| Internal Ridge-estimator reconciliation | documented; 92.3% overlap; not merged or presented |
| Preserved code hashes | `src/pipeline.py` and `strategies.py` unchanged |
| ZIP extraction | verified in a clean temporary directory |

---

## 8. Remaining manual actions

1. Add the team roster to the report title page if required by the course; the package currently uses “DSA5205 Project 2 Team”.
2. Record the 15–25 minute video.
3. Sign off the packaged derived panels against the team's canonical raw snapshot.
4. If the course requires point-in-time index membership, note that the current data source cannot provide it.
5. Review the AI-use declaration against the team's own records.
6. Confirm that the 23-page report satisfies the specific page-count rule of the current marking scheme.
