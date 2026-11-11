# DSA5205 Project 2 — Ridge-Based Cross-Sectional Equity Strategy

**Final submission package**

This project evaluates one systematic trading idea:

> Historical market data → liquidity screening → Top-300 liquid universe → 3×3 momentum–volatility double sorting → 30 candidate stocks → Ridge return prediction → Top-10 stocks → equal-weight portfolio → monthly rebalancing → out-of-sample evaluation net of transaction costs.

Ridge is the **designated primary research model**. LASSO, Random Forest and XGBoost are comparison models on the same prediction task. B1 and B3 are momentum baselines; SPY and QQQ are investable benchmarks. The original exploratory Ridge extension is retained as an exploratory analysis.

The package contains an executed 14-section notebook, a 23-page academic report (PDF and editable LaTeX), a reproducible pipeline, 113 automated tests, and all result tables and figures.

> Academic research and historical backtesting only. No live trading, no investment advice.

---

## 1. Research question and strategy

**Question.** Can an economically motivated momentum–volatility screening strategy, combined with Ridge-based return prediction, generate useful stock rankings and competitive out-of-sample portfolio performance relative to traditional market benchmarks and simpler momentum strategies?

**Eligibility.** At each decision close, a stock must have a close of at least \$5, ADV20 of at least \$10 million (the 20-session mean of close × volume), at least 95% raw price coverage over the 21-, 63- and 252-session windows, and finite values for all fifteen features. The 300 most liquid eligible names by ADV20 form the liquid universe. The 300-name cap is a fixed design choice, not an out-of-sample optimisation; a month with fewer than 90 eligible names would be skipped. In this snapshot every month reached 300.

**3×3 double sort.** Eligible stocks are ranked by `mom_2_12` descending and split into three approximately equal momentum groups. Within each momentum group, stocks are sorted by `vol_63` ascending and split into three approximately equal volatility cells. Each momentum group contributes 3 low-volatility, 4 medium-volatility and 3 high-volatility candidates, i.e. ten names per group and thirty in total. The 3–4–3 split is a fixed, symmetric construction because ten cannot be divided evenly into three cells; the extra candidate is assigned to the middle cell so that neither volatility extreme is favoured. It is not claimed to be optimal.

Within a cell, the high-momentum group takes the highest `mom_2_12` names, the low-momentum group the lowest, and the medium-momentum group the names closest to the cell median. Ties are broken by ticker ascending. A cell that is too small causes the decision date to be skipped rather than filled from outside the cell.

**Portfolio.** The thirty candidates are scored by Ridge. The ten highest predicted returns receive equal 10% target weights and all other candidates receive zero. Trades execute at the next session open and the portfolio rebalances monthly. The primary strategy is fully invested, long-only and unlevered. Transaction costs are 20 basis points per traded side on the sum of absolute weight changes against drifted pre-trade weights.

**Out-of-sample comparison set:**

| Strategy | Role | Selection | Weighting |
|---|---|---|---|
| Ridge | Primary ML model | 3×3 pool, Ridge Top 10 | Equal 10% |
| LASSO | Alternative ML | 3×3 pool, LASSO Top 10 | Equal 10% |
| Random Forest | Alternative ML | 3×3 pool, RF Top 10 | Equal 10% |
| XGBoost | Alternative ML | 3×3 pool, XGB Top 10 | Equal 10% |
| B1 | Momentum baseline | Momentum Top 10, full liquid universe | Equal 10% |
| B3 | Momentum baseline | Momentum Top 10, 3×3 pool | Equal 10% |
| SPY | Investable benchmark | Buy-and-hold S&P 500 ETF | Fully invested |
| QQQ | Investable benchmark | Buy-and-hold Nasdaq-100 ETF | Fully invested |

---

## 2. Model-assessment policy

The project distinguishes four roles:

1. **Primary research model:** Ridge.
2. **Hyperparameter selection:** validation MSE within each model (Ridge selects `alpha = 1000`).
3. **Comparison models:** LASSO, Random Forest and XGBoost on identical data and rules.
4. **Superseded auxiliary artifact:** an earlier `lowest_validation_mse` rule that selected LASSO (`alpha = 0.001`).

Ridge is designated primary because the fifteen predictors are a dense set of correlated financial characteristics, L2 shrinkage is the natural regularisation for that design, Ridge has the highest validation Rank IC (0.0438 versus 0.0250 for LASSO), and rolling-window diagnostics show lower relative coefficient-magnitude change (0.176 versus 0.282). Ridge is **not** claimed to minimise validation MSE: LASSO's MSE is 0.022281 versus Ridge's 0.022422.

The automatic minimum-MSE rule is **not** part of the active workflow. Its files are retained read-only and clearly labelled under:

```
results/validation/archive/superseded_automatic_selection/
```

The active primary-model contract is:

```
results/validation/primary_model_contract.json
results/validation/primary_model_designation.csv
```

**Chronology disclosure.** The original project pre-specified Ridge. A later engineering revision introduced the automatic rule and selected LASSO, and out-of-sample results were reviewed during that revision. This revision restores the Ridge designation on economic and diagnostic grounds, but it is **not a clean pre-registration**. The contract records `clean_preregistration: false` and `designation_informed_by_prior_oos_review: true`. No test-period metric was used to choose Ridge or its alpha.

---

## 3. Internal identifier note (reproducibility only)

The extension's internal result table retains the historical identifier `R4_RidgeDoubleSort30` for reproducibility. It is **not** a separate strategy: it applies the same economic rule as Main Ridge (3×3 screen, Ridge predicted-return Top 10, equal weights) but uses a different fitted Ridge estimator, trained on the extension's 60-candidate pool (8,520 refit rows) rather than the main 30-candidate panel (4,260 refit rows). The two results are therefore not merged: the authoritative primary OOS result is Main Ridge at 43.0% CAGR; the extension estimator's 44.2% is retained only in the underlying audit files and is not presented as a second Main Ridge result.

The reconciliation is documented for reproducibility in:

```
results/extensions/ridge_estimator_reconciliation.json
results/extensions/ridge_estimator_holdings_overlap.csv
```

Mean holdings overlap is 92.3%, 18 of 48 months have identical holdings, and the monthly return correlation is 0.9975. These files support the technical note; they are not used in the eight-strategy OOS ranking or the report.

---

## 4. Results at a glance

**Validation 2020–2021** (22 settled cohorts, post-cost):

| Model | MSE | Rank IC | Sharpe | CAGR | Max DD |
|---|---:|---:|---:|---:|---:|
| Ridge | 0.02242 | 0.0438 | 0.918 | 45.7% | −31.7% |
| LASSO | 0.02228 | 0.0250 | 0.926 | 46.1% | −30.0% |
| Random Forest | 0.02241 | 0.0261 | 0.866 | 40.2% | −33.2% |
| XGBoost | 0.02264 | −0.0098 | 0.693 | 26.6% | −29.2% |

**Out-of-sample 2022–2025** (48 monthly cohorts, net of 20 bps per side):

| Strategy | CAGR | Volatility | Sharpe | Max DD | Turnover |
|---|---:|---:|---:|---:|---:|
| Ridge | 43.0% | 32.6% | 1.27 | −28.5% | 1.26 |
| LASSO | 42.1% | 32.3% | 1.26 | −28.2% | 1.29 |
| Random Forest | 40.0% | 32.8% | 1.20 | −29.0% | 1.25 |
| XGBoost | 38.5% | 30.8% | 1.22 | −27.1% | 1.32 |
| B1 | 73.1% | 37.7% | 1.66 | −19.2% | 0.65 |
| B3 | 51.5% | 29.0% | 1.59 | −15.0% | 0.94 |
| SPY | 12.7% | 15.3% | 0.86 | −19.7% | 0.02 |
| QQQ | 14.8% | 19.9% | 0.80 | −25.6% | 0.02 |

**Interpretation.** Ridge comfortably beats SPY and QQQ after costs and has the highest OOS Sharpe and CAGR among the four ML models. It does **not** beat simple momentum. Within the same candidate pool, B3's momentum ranking beats the Ridge ranking by 8.5 percentage points of CAGR and 0.32 of Sharpe, with lower turnover and a shallower drawdown. Main Ridge versus B3 is the direct ranking comparison; Main Ridge versus B1 changes both the universe and the ranking, so it is a comparison of complete strategies. The report states this without re-selecting a strategy after seeing the test period.

---

## 5. Final directory structure

```text
DSA5205_Project2_Final/
├── README.md
├── requirements.txt
├── CHANGELOG.md
├── INTEGRATION_AUDIT.md
├── COMPLIANCE_CHECKLIST.md
├── SUBMISSION_MANIFEST.md
├── run_pipeline.py
├── run_tests.py
├── config/
│   ├── experiment_config.yaml
│   ├── pipeline.json
│   └── preserved_sources.json
├── src/
│   ├── pipeline.py            # preserved verbatim
│   ├── backtest.py            # original rules + comparison helpers
│   ├── qa.py, reporting.py, visualize.py
│   ├── data_pipeline.py
│   ├── features.py
│   ├── models.py
│   ├── validation.py
│   ├── metrics.py
│   ├── model_selection.py     # legacy auxiliary module, not in active workflow
│   ├── research_policy.py     # Ridge primary-model designation
│   ├── diagnostics.py
│   ├── final_refit.py
│   ├── oos_evaluation.py
│   ├── oos_comparison.py
│   ├── score_position.py
│   ├── reconciliation.py
│   ├── ridge_extension.py
│   ├── visualization.py
│   └── utils.py
├── strategies.py              # preserved verbatim
├── notebooks/
│   └── Project2_Final.ipynb
├── data/
│   ├── processed_ready/       # model-ready panels (required)
│   └── derived/               # compact extension candidate panel (required)
├── results/
│   ├── validation/
│   │   ├── figures/
│   │   └── archive/superseded_automatic_selection/
│   ├── oos/
│   │   ├── eight_strategy/
│   │   └── figures/
│   ├── diagnostics/
│   ├── extensions/
│   │   └── figures/
│   └── models/refit/
├── figures/                    # curated report figures (Figures 1–8, A1–A2)
├── report/
│   ├── Project2_Report.pdf
│   ├── Project2_Report.tex
│   ├── build_report.py
│   └── generated/
└── tests/
```

---

## 6. Installation

```bash
cd DSA5205_Project2_Final
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name dsa5205-project2 --display-name "Python (DSA5205)"
```

A LaTeX installation (`pdflatex`, `booktabs`, `caption`, `natbib`, `fancyhdr`, `titlesec`, `enumitem`, `setspace`, `float`) is needed only to rebuild the report.

Recorded environment: Python 3.11.4, numpy 1.26.4, pandas 2.2.3, pyarrow 18.1.0, scipy 1.17.1, scikit-learn 1.9.1, xgboost 3.2.0, matplotlib 3.11.2, PyYAML 6.0, joblib 1.6.0, pypdf 7.0.1, pytest 7.4.0, nbformat 5.7.0, nbclient 0.5.13.

---

## 7. Reproduction commands

```bash
# 1. Validate data, splits and the realised-label embargo
python run_pipeline.py --stage data

# 2. Train the four models on 2010-2019 and tune on 2020-2021
python run_pipeline.py --stage validation

# 3. Freeze Ridge as the primary research model
python run_pipeline.py --stage designate --approve

# 4. Refit Ridge and the three comparison models on 2010-2021
python run_pipeline.py --stage refit

# 5. Four-model out-of-sample evaluation, 2022-2025
python run_pipeline.py --stage oos

# 6. Score-to-position illustration (real 2025-06-30 predictions)
python run_pipeline.py --stage score

# 7. Preserved exploratory extension and internal reconciliation
python run_pipeline.py --stage extension

# 8. Eight-strategy reconciliation and Figures 1-6
python run_pipeline.py --stage eight

# 9. Tests (add --notebook to execute the notebook)
python run_tests.py
python run_tests.py --notebook

# 10. Rebuild the report
python report/build_report.py
```

One-command run:

```bash
python run_pipeline.py --stage all --approve
```

Without `--approve`, `--stage all` stops after validation with exit code `3`. There is no active automatic model-selection stage.

---

## 8. Notebook

`notebooks/Project2_Final.ipynb` contains the 14 required sections:

1. Research Question and Economic Motivation
2. Markets, Assets, Trading Calendar and Benchmarks
3. Data Sources, Cleaning and Universe Construction
4. Labels, Features and Timing Alignment
5. Feature Correlation and Ridge Model Justification
6. Training, Validation and Hyperparameter Tuning
7. Four-Model Validation Comparison
8. Primary Ridge Strategy and Trading Rules
9. Final OOS Evaluation: Four ML Models
10. Final OOS Evaluation: Ridge vs B1, B3, SPY and QQQ
11. Interpretation and Limitations
12. Exploratory Ridge Strategy Extensions
13. Conclusion
14. Reproducibility and AI-Use Declaration

It is executed end-to-end (40 cells, 0 errors, 14 embedded figures) and changes to the project root automatically.

---

## 9. Academic report

`report/Project2_Report.pdf` — 23 pages, within the 15–25 page requirement.
`report/Project2_Report.tex` — editable LaTeX source.
`report/build_report.py` — rebuilds tables, `.tex` and `.pdf` from `results/`.

Figures: Figure 1 score-to-position; Figures 2–7 the six out-of-sample exhibits; Appendix Figures A1–A2 the correlation and coefficient-stability diagnostics. All figure and table labels are referenced in the text.

---

## 10. Data and reproducibility

Included: `data/processed_ready/` model-ready panels and `data/derived/extension_top300_labeled.parquet`. Raw vendor data (about 57 MB) and the full feature panel (about 331 MB) are excluded to keep the submission small.

To rebuild from raw vendor data:

```bash
mkdir -p data/raw
cp <original_project>/data/raw/market_data.parquet data/raw/
python -m src.data_pipeline --rebuild-panel --frequency monthly
```

Data source: Yahoo Finance via `yfinance`; current S&P 500 constituents from Wikipedia. The active stock universe is **survivorship- and constituent-selection-biased** because membership is today's list, not point-in-time; this may overstate active strategy performance relative to SPY and QQQ, whose buy-and-hold returns come directly from historical ETF prices. Vendor adjustments may be revised. Neither limitation is corrected, and the direction or magnitude of the survivorship effect is not precisely measured.

---

## 11. Tests

```bash
python run_tests.py --notebook
```

**115 passed, 0 failed** in the final verification (114 fast tests + notebook execution).

Coverage: data integrity, leakage, four-model fairness, legacy selection module, Ridge policy, backtest rules, refit, extension preservation, eight-strategy reconciliation, score-to-position, internal estimator reconciliation, reproducibility and report integrity.

---

## 12. Known limitations

1. Survivorship and constituent look-ahead bias (not corrected).
2. Vendor data may be revised; not exchange-grade point-in-time data.
3. Small test sample: 48 monthly observations with overlapping holding periods.
4. Validation selection optimism: the same 2020–2021 window selects hyperparameters and compares models.
5. Static model through the test period; no rolling retraining.
6. Constant 20 bps cost; no market impact, borrow, taxes or capacity.
7. Full equity beta (1.64) with no risk overlay; deep drawdowns.
8. Zero risk-free rate for Sharpe, Sortino and alpha.
9. Jensen alpha against SPY only; no Fama–French–Carhart attribution.
10. The Ridge designation is not a clean pre-registration.
11. The exploratory extension was designed after earlier OOS review and is not independent confirmation.
12. Only monthly frequency is evaluated in depth.

---

## 13. Provenance and AI use

The strategy, data pipeline, features, labels, backtest, benchmarks and exploratory extension come from `DSA5205_Project2_Pipeline`. The four-model factories and default grids also originate there. This revision adds the Ridge-centred research policy, eight-strategy reconciliation, diagnostics integration, score-to-position evidence, tests, notebook and report.

The teammate submission was inspected for its four-model outputs and reproducibility record. Its rewritten `strategies.py` was **not** adopted; the original extension is preserved byte-for-byte and hash-locked. See `INTEGRATION_AUDIT.md` for the file account and `CHANGELOG.md` for the history.

AI-use declaration: this revision was prepared with AI assistance (DeepSeek Harness) for code restructuring, test generation, diagnostic integration, figure production and LaTeX drafting. All numbers were produced by executing the code. Full declaration in report §15 and notebook §14.
