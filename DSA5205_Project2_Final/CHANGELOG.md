# Changelog

All significant changes to the integrated DSA5205 Project 2 repository.

---

## [4.0.0] — 2026-11-11 — Final consistency revision

### Narrative and naming

- **Ridge selection narrative corrected.** Removed the claim that the primary model was “designated before the test period is evaluated”. The report now states that model assessment is based on economic considerations and validation-stage evidence, and that OOS performance is reported separately and is not used as a numerical model-selection criterion. The chronology disclosure (project history includes prior OOS review; not a clean pre-registration) is retained.
- **Survivorship-bias statements corrected.** The report no longer claims that survivorship bias affects every strategy including the benchmarks. It now separates (i) survivorship/constituent-selection bias in the active stock universe from (ii) vendor price-adjustment and corporate-action effects, and notes that SPY/QQQ returns come directly from historical ETF prices.
- **Former R4 removed from all user-facing research materials.** The report, notebook, figures, tables and README do not present R4 as a separate strategy, variant or benchmark. The internal identifier `R4_RidgeDoubleSort30` is retained only in source code and underlying audit files; the mapping and estimator difference (4,260 vs 8,520 refit rows; 43.0% vs 44.2%) are documented in a technical note.
- **Extension presentation reduced to the genuine modifications.** Public extension tables and figures now contain B1, B2, B3, R1, R2, R3, SPY and QQQ (eight rows), excluding the former R4 sleeve. Internal nine-row records remain for reproducibility.
- **Ridge/LASSO OOS interpretation sharpened.** The report states Ridge's marginally better OOS MSE, CAGR and Sharpe under the original four-model experiment, explicitly notes that the differences are not tested for statistical significance, and avoids claims of universal superiority or proof that LASSO's sparsity is inferior.
- **AI-use declaration finalized.** Removed the request for team members to review the declaration. The factual statement covers code restructuring, test generation, diagnostic integration, figure production and LaTeX drafting; no experiment is attributed to AI.

### Report and outputs

- Rebuilt the report at **23 pages**, Figures 1–7 (score-to-position; six OOS exhibits) and Appendix Figures A1–A2, 12 tables. No undefined references and no overfull boxes.
- Removed the R4 reconciliation subsection and Figure 8; renamed internal reconciliation outputs to `ridge_estimator_reconciliation.json` / `ridge_estimator_holdings_overlap.csv` and `fig_internal_estimator_reconciliation.png`.
- Extension figures renamed to `extension_cumulative_wealth.png`, `extension_annual_returns.png`, `extension_drawdowns.png` and regenerated from the eight public strategies.

### Tests and packaging

- Added naming/publication tests: public extension table has eight rows and excludes the internal sleeve; report LaTeX and notebook sources contain no `R4` strategy presentation.
- Full suite: **115 passed, 0 failed**, including notebook execution (40 cells, 0 errors, 14 embedded figures).
- Updated `README.md`, `INTEGRATION_AUDIT.md`, `COMPLIANCE_CHECKLIST.md`, `SUBMISSION_MANIFEST.md` and the final ZIP.

### Preserved

- All empirical results, model parameters, features, universe construction, trading rules, costs, benchmarks and exploratory strategies are unchanged.
- The former R4 internal results are preserved for audit, not merged or averaged.
- `src/pipeline.py` and `strategies.py` remain byte-identical and hash-locked.

---

## [3.0.0] — 2026-11-11 — Final submission refinement

### Project and narrative

- Renamed and reorganised the working project to `DSA5205_Project2_Final/` with a clean `results/{validation,oos,diagnostics,extensions}` structure, a top-level `figures/` directory and `report/Project2_Report.{pdf,tex}`.
- Final narrative fixed around one research question: momentum–volatility screening → Ridge ranking → Top-10 equal-weight portfolio → OOS evaluation. The same eight-step strategy description is used in the notebook, report, figures and README.
- Added explicit documentation of the Top-300 liquidity universe and the dependent 3×3 double sort, including the exact 3–4–3 allocation, within-cell selection rules and tie-breaking.

### Model assessment

- Removed the superseded `lowest_validation_mse` rule from the active CLI and notebook workflow. There is no active `select` stage.
- The Ridge designation remains in `primary_model_contract.json`; the historical LASSO output is preserved read-only under `results/validation/archive/superseded_automatic_selection/` with a README and manifest marking it as a superseded auxiliary artifact.
- `src/model_selection.py` is retained only for audit/reproduction and is explicitly labelled legacy.
- Validation, refit and OOS use `active_contract`, which now requires the Ridge designation and has no automatic fallback.
- Added precise training-method documentation: Ridge uses scikit-learn `solver='auto'` (Cholesky for the dense problem); LASSO uses cyclic coordinate descent; Random Forest uses bagged trees; XGBoost uses histogram gradient boosting with squared error.

### New evidence

- New `src/score_position.py` and Figure 1: a real 30 June 2025 score-to-position example with thirty actual Ridge scores, the Top-10 cut-off and the 10%/0% target weights.
- New `src/reconciliation.py`: Main Ridge and R4 are confirmed to implement the same economic rule but use different fitted models (4,260 vs 8,520 refit rows; 30- vs 60-candidate panels). Mean holdings overlap 92.3%, identical holdings in 18 of 48 months, return correlation 0.9975, CAGR difference 1.17 pp. The records are documented and **not merged**; R4 is excluded from the eight-strategy ranking and appears only as a main-rule reference.
- Added corporate-action and transaction-cost verification notes to the report: adjusted opens used consistently, no manual override, and the cost formula charges both purchases and sales.

### Report

- Revision 3.0 report layout (superseded by revision 4.0: 23 pages, Figures 1–7, Appendix A1–A2).
- Reframed the historical automatic selection as a superseded auxiliary artifact; removed any implication that it is the active model-assessment policy.
- Added the 3×3/3–4–3 explanation, exact training methods, the R4 reconciliation and the score-to-position exhibit.
- Compiles with no undefined references and no overfull boxes.

### Tests and packaging

- Added `test_final_workflow.py` (score-to-position, R4 reconciliation, no active auto-selection, archive labelling).
- Updated the full suite for the final paths, Ridge designation and report filenames.
- Full suite at revision 3.0: **113 passed, 0 failed**, including notebook execution from a clean kernel.
- Added `SUBMISSION_MANIFEST.md` with an actual-path file inventory.

### Preserved

- `src/pipeline.py` and `strategies.py` remain byte-identical to the authoritative pipeline; the original `portfolio_backtest` and `performance` bodies are hash-locked.
- All original experiment results are retained; the superseded selection files were moved, not deleted.
- The original project was backed up outside the final directory before restructuring.

---


## [2.0.0] — 2026-11-11

### Research design

- **Ridge designated as the primary research model** on economic and diagnostic grounds, replacing the automatic `lowest_validation_mse` rule as the mechanism that defines the primary strategy.
- New `src/research_policy.py`: explicit Ridge-centred policy, rationale, validation-based hyperparameter choice within Ridge, and a contract that discloses that the designation is not a clean pre-registration.
- The historical LASSO minimum-MSE selection is **preserved and archived** under `results/validation/archive/`. The original `selection_contract.json` is not overwritten. The two contracts are reported side by side.
- `src/final_refit.py` and `src/oos_evaluation.py` now use the active contract (Ridge when present, historical rule as fallback).

### Eight-strategy out-of-sample comparison

- New `src/oos_comparison.py`: reconciles the main experiment (Ridge, LASSO, RF, XGBoost), the preserved extension (B1, B3) and the benchmarks (SPY, QQQ) on the same 48 decision dates.
- Verifies date alignment, exact SPY/QQQ equality across the two backtest engines, and that B3 is the momentum Top-10 of the main 30-candidate pool, date by date.
- Produces `results/oos/eight_strategy/` tables and the six required figures (`fig1`–`fig6`) with publication formatting.
- Documents turnover, transaction costs, gross-versus-net decomposition, beta and Jensen alpha for all eight strategies.

### Diagnostics

- New `src/diagnostics.py`: loads and re-verifies the supplied correlation and coefficient-stability diagnostics against the training slice.
- Confirmed 12 feature pairs with |r| ≥ 0.70; Ridge mean relative coefficient change 0.176 versus LASSO 0.282 (37.4% lower); LASSO zero-coefficient fraction 40.1%; Ridge 50 sign flips versus LASSO 1 across 59 rolling windows.
- Produces compact report figures for the correlation matrix and stability comparison.

### Notebook

- Restructured into the required 14 sections, from research question to reproducibility and AI-use declaration.
- New `notebooks/Project2_Ridge_Primary_Study.ipynb`, executed end to end (37 cells, 0 errors, 13 embedded figures). Superseded in revision 3.0 by `Project2_Final.ipynb`.
- The old `Project2_Four_Model_Complete.ipynb` is retained for history but is superseded by the new notebook.

### Academic report

- New `report/` directory: PDF, editable LaTeX source, build script, generated tables and figures (later revised to 23 pages in 4.0).
- Fifteen numbered sections, 12 tables, Figures 1–6 for the six required OOS exhibits, Appendix Figures A1–A2 for diagnostics, and 20 verified references.
- All numbers are generated from `results/` at build time; the report script checks the page count.
- Includes the chronology disclosure, limitations, reproducibility commands and an AI-use declaration.

### Tests

- Added `test_research_policy.py` (7 tests), `test_eight_strategy.py` (9 tests) and `test_report.py` (9 tests).
- Updated `test_reproducibility.py` for the new notebook, report and eight-strategy outputs.
- Full suite: **106 passed, 0 failed**, including notebook execution from a clean kernel.

### Data and provenance

- Preserved `src/pipeline.py`, `strategies.py`, the original `portfolio_backtest` and `performance` bodies, and the original extension results.
- No trading rule, universe definition, feature, label, cost assumption or benchmark was changed.

### Fixed

- Benchmark-relative beta/alpha/tracking-error alignment now joins on decision dates rather than row indices.
- Eight-strategy OOS reconciliation verifies cross-engine benchmark equality and the B3 rule before producing any table or figure.
- `run_pipeline.py` now has explicit `designate`, `eight` and legacy `select` stages; `all` stops without `--approve`.

---


## [1.0.0] — 2026-11-11

### Added

**Four-model validation stage**
- `src/validation.py`: 23-configuration tuning across Ridge (6), LASSO (5), Random Forest (4) and XGBoost (8), fitted on 2010–2019 and scored on 2020–2021.
- Deterministic validation-MSE selection with grid-index tie-breaking; optional validation-Sharpe rule.
- Validation predictions with decision, execution and label-end dates, prediction ranks and identifiers.
- Six required tables: `validation_prediction_metrics.csv`, `validation_portfolio_metrics.csv`, `validation_hyperparameter_search.csv`, `validation_model_comparison.csv`, `validation_monthly_returns.csv`, `validation_predictions.csv`.
- Supporting tables: `split_summary.csv`, `label_embargo_excluded.csv`, `data_quality.json`, `validation_best_params.json`, `validation_run_metadata.json`.

**Metrics**
- `src/metrics.py`: MSE, RMSE, MAE, ordinary sklearn \(R^2\), training-mean benchmark-relative OOS \(R^2\), raw and cross-sectional directional accuracy, Pearson correlation, pooled and per-month Spearman rank IC, rank-IC positive share.
- Portfolio metrics: gross/net total return, CAGR, average and median monthly return, annualised volatility, Sortino, Sortino, maximum drawdown, Calmar, turnover, total and average transaction cost, positive-month share, best/worst month, benchmark-relative annualised return, tracking error, information ratio, beta and Jensen alpha.
- Explicit zero-risk-free labelling of Sharpe, Sortino and alpha.

**Model selection contract**
- `src/model_selection.py`: `lowest_validation_mse` (default) and `highest_validation_sharpe` rules.
- Approval gate: `SelectionNotApproved` without `--approve` / `APPROVE_FINAL_SELECTION=True`.
- `model_selection_summary.csv` and `selection_contract.json` with primary model, criterion, selected model, hyperparameters, validation period, timestamp, run id, validation fingerprint, `test_used_for_selection: false` and `test_data_accessed: false`.
- Idempotent reuse: a re-run with an unchanged validation fingerprint preserves the original selection timestamp and refreshes `last_verified_utc`.
- `model_selection_pending.json` written when approval is withheld.

**Final refit**
- `src/final_refit.py`: refit of the selected model and the three pre-planned comparators on all eligible 2010–2021 rows (`label_end_date <= 2022-01-01`).
- `final_refit_summary.csv`, `final_model_metadata.json`, and serialized `results/models/refit/{ridge,lasso,rf,xgb}.joblib` with SHA-256 artifact hashes.

**OOS evaluation**
- `src/oos_evaluation.py`: 2022–2025 prediction and portfolio evaluation for all four refitted models.
- `oos_prediction_metrics.csv`, `oos_portfolio_metrics.csv`, `oos_model_comparison.csv`, `oos_monthly_returns.csv`, `oos_predictions.csv`, `validation_to_oos_degradation.csv`, `oos_run_metadata.json`, `final_selection_audit.json`.
- `oos_run_metadata.json` records `selection_changed_after_test: false` and the selected model's descriptive OOS ranks.

**Visualization**
- `src/visualization.py`: 15 validation figures (MSE bar, RMSE/MAE, Sharpe, CAGR, volatility, risk–return, wealth, drawdown, monthly heatmap, prediction heatmap, trading heatmap, MSE-vs-Sharpe, turnover/cost, ranking summary, benchmark comparison) and 5 OOS figures (wealth, drawdown, annual returns, benchmark comparison, validation-to-OOS degradation).
- All figures read saved CSVs; no performance number is hard-coded.

**Preserved exploratory extension (nine internal sleeves)**
- `src/ridge_extension.py`: driver that calls the unmodified `strategies.py` functions (`pools`, `train_ridge`, `targets_for_date`, `simulate`, `performance`).
- Outputs: `strategy_summary.csv`, `period_returns.csv`, `annual_returns.csv`, `holdings.csv`, `trades.csv`, `validation_grid.csv`, `fit_info.json`, `prediction_metrics.json`, `ridge_test60_predictions.csv`, `ridge_rank_ic.csv`, `screening_audit.csv` and five figures.
- Byte-identical `strategies.py` and `src/pipeline.py`; hash-locked in `config/preserved_sources.json`.

**Data and configuration**
- `config/experiment_config.yaml`: single source of truth for dates, embargo policy, grids, portfolio rules, selection rule and output paths.
- `src/data_pipeline.py`: model-ready validation, chronological splits with realized-label embargo, extension candidate panel builder and a `--rebuild-panel` path from raw data.
- `src/features.py`: feature registry, descriptions and preprocessing contract.
- `src/utils.py`: hashing, UTC timestamps and environment metadata.
- `data/derived/extension_top300_labeled.parquet`: 57,600-row compact candidate panel (192 dates × 300 names) with preserved labels, so the extension can run without the 331 MB feature panel.

**Tests**
- Eight test modules, all executed: data integrity, leakage, four models, selection, backtest, refit, extension preservation and reproducibility.
- Notebook execution test from a clean kernel (`RUN_NOTEBOOK_TEST=1`).
- Golden-fixture regression for the extension and source/function hash locks.

**Documentation**
- `README.md` with installation, exact commands for every stage, output inventory, current results, reproducibility and limitations.
- `INTEGRATION_AUDIT.md` with file provenance, reuse decisions, leakage controls, test results and open items.
- `CHANGELOG.md` (this file).

**Notebook**
- `notebooks/Project2_Four_Model_Complete.ipynb`: 49 cells, 24 required sections, executed end-to-end with 0 errors.
- Portable relative paths; changes to the project root automatically; no absolute local paths.

### Changed

- `src/backtest.py`: appended four-model helper functions. The original `portfolio_backtest` and `performance` bodies are unchanged and hash-locked.
- `src/models.py`: rewritten as the extended four-model module. Ridge/LASSO/RF grids unchanged; XGBoost `n_estimators` is now tuned over `{100, 200}` instead of fixed at 200, so the grid covers `max_depth`, `learning_rate` and `n_estimators` as required. The legacy `run_model` entry point is retained.
- Benchmark-relative metrics now align strategy and benchmark series on `decision_date` rather than the DataFrame row index. This fixed a bug where beta, alpha, tracking error and information ratio were `NaN`.

### Preserved unchanged

- `src/pipeline.py` (data, features, screening, labels).
- `strategies.py` (exploratory extension, nine internal sleeves).
- Universe, screening, target, portfolio construction, execution timing, 20 bps cost, SPY/QQQ benchmarks.
- `config/pipeline.json`.

### Known issues carried forward

- `src/pipeline.py` writes a stale `model_contract.json` with `2021-01-01` / `2023-01-01` split boundaries. The integrated pipeline does not read it. See `INTEGRATION_AUDIT.md` §11.
- Survivorship and constituent look-ahead bias remain; they are documented, not corrected.

---

## Verification record

- `RUN_NOTEBOOK_TEST=1 python3 -m pytest -q` → **115 passed, 0 failed**.
- Notebook executed from a clean kernel → **40 cells, 0 errors, 14 embedded figures**.
- Report compiled to 23 pages with no undefined references and no overfull boxes.
- Nine-strategy extension CAGRs reproduce the teammate reference to the last displayed digit.
- Validation and OOS tables regenerated from the packaged snapshot; primary-model contract fingerprint verified after notebook re-execution.
