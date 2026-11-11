# Integration Audit — DSA5205 Project 2

**Date:** 2026-11-11 (Asia/Singapore)
**Revision:** 3.0 — final research refinement, technical verification and submission packaging
**Authoritative source:** `DSA-5205/DSA5205_Project2_Pipeline.zip` → `DSA5205_Project2_Pipeline/`
**Reference source:** `DSA-5205/Project2_Submission.zip` → nested `Group_X_Source_Code.zip` → `DSA5205_Project2/`
**Supplementary diagnostics:** `results/diagnostics/` (from `research_diagnostics.zip`)
**Backup of the pre-final project:** `../_archive/DSA5205_Project2_Integrated_pre_final/` (outside the submission directory)

---

## 1. Files that came from the original Pipeline

Copied **byte-for-byte** from the authoritative pipeline:

| File | Role | SHA-256 |
|---|---|---|
| `src/pipeline.py` | Data acquisition, cleaning, feature engineering, 3×3 screening, next-open labels | `f0727c83cacaf83d5e48d290e672ab78feaa0c2cad3a4763e7081e28ca5176db` |
| `strategies.py` | Nine-strategy Ridge extension (B1–B3, R1–R4) | `224e7ad143e27cb4d07cb83e81699eead2e5548e7b0e803aa62a30ae4d50fe6a` |
| `src/qa.py`, `src/reporting.py`, `src/visualize.py` | Original QA and reporting helpers | unchanged copies |
| `config/pipeline.json` | Original pipeline parameters | unchanged copy |
| `data/processed_ready/{monthly,quarterly,semiannual}/*` | Model-ready panels and screening audit | copied |

`src/backtest.py` is a copied original file with appended comparison helpers. The two original functions are unchanged; their source hashes are recorded in `config/preserved_sources.json` and asserted by `tests/test_backtest.py`:

- `portfolio_backtest`: `4969af1014262c01a3a8bd408ad712247b36c01ce9b09d674ee47cb6d7a7c9d8`
- `performance`: `531310535b2105f6e550c3d89fdd5094af25e873b7796d606460926b8b8c5e32`

The original notebooks were read as behavioural references. Their extension logic is reproduced by `src/ridge_extension.py`.

---

## 2. Reference submission: reuse and non-reuse

| Item | Use |
|---|---|
| `Results/four_model/validation_model_comparison.csv` | External check on validation MSE and parameters |
| `Results/four_model/test_*_comparison.csv` | External check on OOS numbers |
| `Results/ridge_extension/strategy_summary.csv` | External check on the exploratory extension (nine internal sleeves) |
| `notebooks/03_model_comparison_and_ridge_strategies.ipynb` | Confirmed the four-model + extension separation |
| `REPRODUCIBILITY_RECORD.md` | Confirmed the expected data shape and reference values |

**Not adopted:** the teammate's rewritten `strategies.py`. It changes pool structure and the `capped` return type. The original `strategies.py` is the authority and is preserved byte-for-byte. The two were compared on a deterministic 300-ticker fixture; the original is the regression baseline.

**Comparison against reference results**

| Quantity | Final result | Reference | Verdict |
|---|---:|---:|---|
| Extension R1 CAGR | 0.8788683515930791 | 0.8788683515930791 | exact |
| Extension B1 CAGR | 0.7312851953949475 | 0.7312851953949475 | exact |
| Extension B3 CAGR | 0.5151040491936363 | 0.5151040491936363 | exact |
| Extension SPY CAGR | 0.12717244601778943 | 0.12717244601778943 | exact |
| LASSO OOS Sharpe | 1.259494 | 1.259494 | exact |
| LASSO OOS CAGR | 0.421480 | 0.421480 | exact |
| Ridge validation MSE | 0.0224217897 | 0.0224226526 | Δ ≈ 8.6e-7 |
| Ridge OOS MSE | 0.0150566322 | 0.0150556264 | Δ ≈ 1.0e-6 |

The extension and LASSO path reproduce exactly, verifying data and trading rules. Ridge/RF differ at the 1e-6 MSE level, consistent with a different intermediate build or a different stored reference run; no rule, grid or selected hyperparameter is affected. The raw parquet SHA-256 differs while row count, tickers and dates match.

---

## 3. Final project structure and cleanup

The pre-final project was backed up to `../_archive/DSA5205_Project2_Integrated_pre_final/` before restructuring. The final tree:

```text
DSA5205_Project2_Final/
├── README.md, requirements.txt, CHANGELOG.md, INTEGRATION_AUDIT.md
├── COMPLIANCE_CHECKLIST.md, SUBMISSION_MANIFEST.md
├── run_pipeline.py, run_tests.py
├── config/{experiment_config.yaml, pipeline.json, preserved_sources.json}
├── src/ (22 modules)
├── notebooks/Project2_Final.ipynb
├── data/{processed_ready, derived}
├── results/{validation, oos, diagnostics, extensions, models}
├── figures/ (curated report figures)
├── report/{Project2_Report.pdf, Project2_Report.tex, build_report.py, generated/}
└── tests/ (12 modules)
```

Renames applied: `results/ridge_extension` → `results/extensions`; `results/research_diagnostics` → `results/diagnostics`; main notebook → `Project2_Final.ipynb`; report → `Project2_Report.*`; report figures moved to top-level `figures/`. All configuration paths, modules, tests, notebook and report builder were updated. Removed from the submission tree: the old 49-cell notebook archive, `outputs/` legacy run_model output, duplicate ZIPs, `__pycache__`, `.pytest_cache`, `.DS_Store`, and LaTeX aux/log files. None of these are required for reproducibility.

---

## 4. Research policy: Ridge primary, automatic rule removed from active workflow

| Role | Implementation |
|---|---|
| Primary research model | Ridge, `results/validation/primary_model_contract.json` |
| Hyperparameter selection | validation MSE within each model; Ridge `alpha=1000` |
| Comparison models | LASSO, Random Forest, XGBoost |
| Superseded auxiliary artifact | LASSO `alpha=0.001`, archived read-only |

The previous `lowest_validation_mse` rule was an engineering convenience, not the research team's model-assessment policy. It has been removed from the CLI (`--stage select` no longer exists), from the notebook workflow and from `active_contract`. `src/model_selection.py` is retained only for audit and is explicitly labelled legacy. Its output is preserved under `results/validation/archive/superseded_automatic_selection/` with a README and `ARCHIVE_MANIFEST.json` stating `active_workflow_uses_this: false`.

The Ridge designation is based on the economic argument, the correlation structure, the validation Rank IC and the rolling coefficient-stability diagnostics. Ridge's `alpha` is still selected by validation MSE within Ridge. No test metric enters the choice.

**Chronology disclosure.** The original project pre-specified Ridge. A later engineering revision introduced the automatic rule and reviewed OOS results. The present designation restores the original pre-specification but is not a clean pre-registration. The contract records `clean_preregistration: false` and `designation_informed_by_prior_oos_review: true`.

---

## 5. Internal identifier note: Main Ridge and `R4_RidgeDoubleSort30`

The extension's internal result table retains the historical identifier `R4_RidgeDoubleSort30` for reproducibility. It applies the same economic rule as Main Ridge (3×3 screen, Ridge predicted-return Top 10, equal weights) but uses a different fitted Ridge estimator:

| Item | Main Ridge | Internal extension estimator |
|---|---|---|
| Training panel | 30-candidate main panel | 60-candidate extension pool |
| Refit rows | 4,260 | 8,520 |
| Alpha | 1000 | 1000 |
| CAGR | 43.0% | 44.2% |
| Mean holdings overlap | 92.3% | — |
| Identical-holding months | 18 of 48 | — |
| Return correlation | 0.9975 | — |

The difference is not a bug; it reflects a different training sample for the same rule. The authoritative primary OOS result is Main Ridge (43.0%). The extension estimator's result is retained only in `results/extensions/ridge_estimator_reconciliation.json` and `ridge_estimator_holdings_overlap.csv` for reproducibility and is **not presented as a separate strategy**, **not merged**, and **not counted** in the eight-strategy ranking or the report.

## 6. Technical verification performed

| Check | Method | Result |
|---|---|---|
| Data schema and coverage | `load_model_ready`, 192 dates, 30 names/date | pass |
| Duplicate ticker-date | loader and tests | none |
| Non-positive prices | raw snapshot check | none |
| Feature finiteness | loader assertion | pass |
| Training labels realised before cutoff | `chronological_splits` | 118 cohorts, max label 2019-12-02 |
| Validation labels realised before test | embargo | 22 cohorts, max label 2021-12-01 |
| Refit cutoff | `label_end_date <= 2021-12-31` | 142 cohorts, max 2021-12-01 |
| Training-only scaling | `Pipeline(StandardScaler)` fitted on train | test with +100 shifted validation |
| Feature/label timing | `decision < execution < label_end` | pass on all rows |
| Ridge solver | fitted estimator inspection | `solver='auto'` (Cholesky for dense) |
| LASSO solver | fitted estimator inspection | cyclic coordinate descent, max_iter 20000, tol 1e-4 |
| Random Forest | fitted estimator inspection | 250 trees, bootstrap, depth 3, leaf 25, max_features 0.7 |
| XGBoost | fitted estimator inspection | hist, square error, 100 trees, depth 2, lr 0.03, subsample 0.8, colsample 0.8, lambda 10 |
| Buy and sell costs | turnover = Σ|target − pretrade|; first-period cost 0.002 | tests pass |
| Gross/net reconciliation | holdings vs portfolio returns | max error < 1e-10 |
| Benchmark cross-engine equality | SPY/QQQ main vs extension | max difference 0.0 |
| B3 selection rule | date-by-date momentum Top 10 of main pool | exact |
| Corporate actions | adjusted opens used consistently; raw check for splits/dividends | documented; vendor spin-off adjustments remain a limitation |
| Main Ridge vs R4 | holdings/returns/training-sample comparison | explained and documented |
| Notebook | clean-kernel execution | 40 cells, 0 errors |
| Report | 23 pages, 0 undefined refs, 0 overfull boxes | pass |
| Full test suite | `RUN_NOTEBOOK_TEST=1 pytest -q` | 115 passed, 0 failed |

**Corporate-action verification detail.** `adj_open = open × adj_close / close` is used for both entry and exit. A raw-snapshot check shows no non-positive closes or duplicate ticker-dates. Some vendor-adjusted events (spin-offs, special distributions) are not represented in the `stock_splits` field, so adjusted returns can differ from raw-price returns around those events. This is documented as a data limitation; the pipeline does not apply an independent corporate-action reconstruction.

**Transaction-cost verification detail.** The cost is `turnover × 20 bps`, with turnover the sum of absolute weight changes. A 10% sale and a 10% purchase each contribute 0.10, so both sides are charged. The first period has turnover 1.0 and cost 0.002. The second period is drift-only and matches a manual calculation. Tests assert these properties.

---

## 7. Leakage controls

| Risk | Control | Test |
|---|---|---|
| Training on unrealised labels | `label_end_date <= boundary`; removes last two 2019 cohorts | `test_training_labels_settle_before_cutoff` |
| Validation labels settling in 2022 | validation embargo; removes Nov/Dec 2021 cohorts | `test_validation_labels_settle_before_test_start` |
| Scaler fitted on validation/test | scaler inside `Pipeline`, fit on training slice | `test_scaler_is_fitted_on_training_rows_only` |
| Test labels changing tuning | validation stage never loads test | `test_corrupting_test_labels_does_not_change_tuning` |
| Test labels changing the designation | designation uses validation + diagnostics only | `test_selection_contract_declares_no_test_use` |
| OOS without approved designation | `require_primary_designation`; exit code 3 | `test_designation_requires_approval` |
| Refit using unavailable labels | refit filter | `test_refit_excludes_labels_unavailable_at_cutoff` |
| Changed validation after freeze | validation fingerprint | `test_require_primary_designation_detects_changed_validation` |
| Future price in decision | decision < execution < label end | `test_labels_use_next_open_to_next_rebalance_open` |
| Survivorship bias | **not eliminated**; documented | — |

---

## 8. Tests executed

**Command:** `RUN_NOTEBOOK_TEST=1 python3 -m pytest -q`
**Result:** `115 passed, 0 failed`.

| Module | Tests |
|---|---:|
| `test_data_integrity.py` | 10 |
| `test_data_leakage.py` | 9 |
| `test_four_models.py` | 11 |
| `test_model_selection.py` (legacy module) | 8 |
| `test_research_policy.py` | 7 |
| `test_backtest.py` | 9 |
| `test_final_refit.py` | 7 |
| `test_ridge_extension.py` | 19 |
| `test_eight_strategy.py` | 8 |
| `test_final_workflow.py` | 7 |
| `test_reproducibility.py` | 11 |
| `test_report.py` | 9 |

---

## 9. Report audit

- `report/Project2_Report.pdf`: 23 pages, within 15–25.
- `report/Project2_Report.tex`: editable source generated from results by `report/build_report.py`.
- Figure 1 score-to-position; Figures 2–7 OOS; Appendix A1–A2 diagnostics.
- 12 tables; 22 figure/table labels, all referenced in the text.
- 20 references, all cited.
- No undefined references, no overfull boxes.
- Includes the chronology disclosure, limitations, reproducibility commands and AI-use declaration.

---

## 10. Remaining manual actions

1. **Team roster.** The title page uses "DSA5205 Project 2 Team"; add names if the course requires a roster.
2. **Video (15–25 minutes).** Not produced here.
3. **Point-in-time universe.** Survivorship bias cannot be removed with the current data.
4. **Data-snapshot sign-off.** Compare packaged derived panels with the team's canonical raw snapshot if exact reproduction is required.
5. **Reference-number sign-off.** Ridge/RF four-model numbers differ from the teammate table at ~1e-6 MSE.
6. **Statistical inference.** No bootstrap confidence intervals; Sharpe differences are descriptive.
7. **Quarterly/semiannual runs.** Supported but not evaluated in depth.

---

## 11. Discrepancies preserved

1. `src/pipeline.py` writes a stale `model_contract.json` with 2021/2023 split boundaries; the active pipeline does not read it. Flagged, not patched.
2. XGBoost `n_estimators` is tuned over `{100, 200}` rather than fixed at 200; documented.
3. The legacy `run_model` tie-breaks by parameter string; the integrated selector uses grid index. No selected configuration changed.
4. The teammate `strategies.py` was not adopted.
5. Benchmarks receive one entry fee only, not monthly rebalancing costs.
6. `equal_weight_30` is not `B2_Momentum30`; original identifiers retained.
