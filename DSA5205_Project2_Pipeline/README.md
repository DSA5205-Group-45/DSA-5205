# DSA5205 Project 2
## Enhancing Momentum-Based Equity Trading Strategies with Machine Learning

**Project Type:** Systematic Equity Trading / Machine Learning / Historical Backtesting

**Market:** US Equities  
**Data Source:** Yahoo Finance (`yfinance`)  
**Primary Experiment:** Monthly Rebalancing  
**Additional Frequencies:** Quarterly and Semiannual  
**Prediction Target:** `target_forward_return`

---

## 1. Project Overview

This project investigates whether machine learning can enhance traditional momentum-based equity trading strategies through improved stock selection and portfolio allocation.

The core investment hypothesis is that historical price momentum may contain predictive information about future stock returns. Machine learning models are introduced to determine whether additional historical features improve return predictions and portfolio performance.

The project has two complementary research components:

**Main Experiment — Machine Learning Model Comparison**

Compare Ridge Regression, LASSO, Random Forest, and XGBoost using identical candidate stocks, features, prediction targets, time splits, and portfolio construction rules.

**Extended Experiment — Ridge-Based Strategy Comparison**

Evaluate how different momentum screening methods, Ridge-based stock selection, and prediction-based portfolio weighting affect investment performance.

All experiments use historical data only. No live trading is conducted.

---

## 2. Data Acquisition

The pipeline automatically retrieves:

- Current S&P 500 constituents from Wikipedia.
- Historical OHLCV and adjusted prices through `yfinance`.
- SPY and QQQ ETF data for market benchmarks.

**Data period:**

- Raw data begins in 2009.
- Research decision dates cover 2010–2025.
- Additional data through 2026 supports the settlement of longer holding periods.

Stocks do not need to have existed throughout the entire period. Newly listed stocks become eligible once sufficient historical data is available to calculate the required features.

Failed downloads are retried and recorded in a download audit.

**Important limitation:** The universe consists of current S&P 500 constituents with historical prices backfilled through time. It is not a point-in-time historical S&P 500 universe. This introduces survivorship and constituent look-ahead bias.

---

## 3. Data Pipeline

The pipeline consists of the following stages:

### Stage 1 — Raw Market Data

Download and preserve historical market observations, including raw OHLCV and adjusted close prices.

Output:

`data/raw/market_data.parquet`

### Stage 2 — Feature Engineering

Calculate 15 historical features:

| Category | Features |
|---|---|
| Momentum | `mom_21`, `mom_63`, `mom_126`, `mom_2_12` |
| Volatility and Risk | `vol_21`, `vol_63`, `beta_252`, `drawdown_63` |
| Trend | `ma_ratio_50`, `ma_ratio_200`, `rsi_14` |
| Liquidity and Volume | `ln_adv_20`, `volume_ratio_21` |
| Price Range | `range_hl` |
| Relative Momentum | `relative_mom_63` |

Features are constructed using historical information available by the decision date.

Missing-value handling and data-quality rules are documented in the pipeline implementation. The intermediate feature dataset retains missing observations for auditing.

Output:

`data/processed_with_nan/full_feature_panel.parquet`

### Stage 3 — Original Statistical Stock Screening

The original screening framework follows:

1. Apply price, liquidity, historical coverage, and feature validity filters.
2. Select up to 300 eligible stocks ranked by ADV20.
3. Divide stocks into High, Medium, and Low Momentum groups using MOM2–12.
4. Within each Momentum group, divide stocks into Low, Medium, and High Volatility groups.
5. Select 3, 4, and 3 stocks from the respective volatility groups within each momentum group.

The resulting candidate universe contains 30 stocks per valid decision date:

| Momentum Group | Low Vol | Medium Vol | High Vol | Total |
|---|---:|---:|---:|---:|
| High | 3 | 4 | 3 | 10 |
| Medium | 3 | 4 | 3 | 10 |
| Low | 3 | 4 | 3 | 10 |
| Total | 9 | 12 | 9 | 30 |

This is a **double-sorting procedure**, not multiplication of Momentum and Volatility.

The original screening framework is used for the main four-model comparison.

### Stage 4 — Model-Ready Datasets

The pipeline produces datasets for:

- `monthly`
- `quarterly`
- `semiannual`

Each dataset includes historical features, stock identifiers, decision and execution dates, forward return labels, and chronological data splits.

Primary model target:

`target_forward_return`

This is the next holding-period simple return expressed as a decimal. For example, `0.05` represents a 5% return.

SPY-relative and QQQ-relative returns are auxiliary analysis fields, not the primary training target.

---

## 4. Training, Validation and Testing

| Dataset | Period | Purpose |
|---|---|---|
| Training | 2010–2019 | Fit model parameters |
| Validation | 2020–2021 | Select hyperparameters |
| Final Refit | Available Train + Validation observations | Re-estimate final model parameters |
| Testing | 2022–2025 | Out-of-sample evaluation |

Labels must be fully realized before they can be used for fitting or model selection.

The main model comparison uses a static final model during the testing period. Features and selected stocks update at each rebalance, while fitted model coefficients remain fixed.

Rolling-window and expanding-window retraining are separate experimental alternatives, not the default main implementation.

---

## 5. Main Experiment — Four ML Models

Four regression algorithms are evaluated:

1. Ridge Regression (L2 Regularization)
2. LASSO Regression (L1 Regularization)
3. Random Forest Regressor
4. XGBoost Regressor

All models use the original 30-stock candidate dataset, the same 15 features, and the same forward-return target.

The general procedure is:

1. Fit candidate models using Training data.
2. Evaluate predefined hyperparameters on Validation data.
3. Select hyperparameters according to Validation prediction error.
4. Refit using available Training and Validation labels.
5. Predict future returns for the 30 candidate stocks during Testing.
6. Select the 10 stocks with the highest predicted returns.
7. Allocate 10% to each selected stock.
8. Rebalance at the next scheduled execution date and account for trading costs.

Ridge and LASSO use feature scaling fitted on training data. Random Forest and XGBoost use their respective complexity controls.

The primary purpose of this experiment is to isolate the effect of the prediction algorithm.

---

## 6. Ridge Extension — Nine Trading Strategies

The extended Ridge experiment compares three momentum baselines, four Ridge-based strategies, and two ETF benchmarks.

| Code Name | Selection Rule | Weighting |
|---|---|---|
| `B1_Momentum10` | Momentum Top 10 directly from ADV20 Top 300 | Equal 10% |
| `B2_Momentum30` | Momentum Top 30 directly from ADV20 Top 300 | Equal 1/30 |
| `B3_MomentumDoubleSort10` | Original 3×3 screened 30 stocks, then Momentum Top 10 | Equal 10% |
| `R1_RidgeWeighted10` | Same 10 stocks as B1 | Ridge prediction-based, max 20% per stock |
| `R2_RidgeFree60` | High/Mid/Low Momentum 20 stocks each; Ridge selects overall Top 10 | Equal 10% |
| `R3_RidgeQuota343` | Same 60 candidates; Ridge selects 3/4/3 from High/Mid/Low Momentum groups | Equal 10% |
| `R4_RidgeDoubleSort30` | Original 3×3 screened 30 stocks; Ridge selects predicted Top 10 | Equal 10% |
| `SPY` | S&P 500 ETF Buy-and-Hold | Fully invested |
| `QQQ` | Nasdaq-100 ETF Buy-and-Hold | Fully invested |

The four Ridge strategies in this extension use a common Ridge predictor trained on the extension's 60-stock candidate sample. This differs from the original 30-stock main-experiment Ridge training sample.

### Important naming clarification

The original pipeline and main model backtest may still use the following legacy identifiers:

| Legacy Name | Interpretation in the Main Experiment |
|---|---|
| `momentum_top10` | Same selection concept as `B3_MomentumDoubleSort10` |
| `model_top10` | Model-predicted Top 10 from the original 30-stock screening; model identity depends on the run |
| `equal_weight_30` | Equal weighting of all 30 original double-sorted candidates |

**These are not all interchangeable with the nine-strategy identifiers.**

In particular:

- `equal_weight_30` is not the same as `B2_Momentum30`.
- `model_top10` is not always Ridge; it can refer to LASSO, Random Forest, or XGBoost.
- `B3_MomentumDoubleSort10` and `R4_RidgeDoubleSort30` share the same screening design but use different final ranking rules.

The original identifiers are retained in the source code for compatibility.

---

## 7. Main Research Comparisons

**B1 vs R1 — Portfolio Allocation**

Does Ridge prediction-based weighting improve returns when the selected stocks are identical?

**B3 vs R4 — Stock Selection**

Does Ridge improve stock ranking compared with traditional momentum within the same 30-stock candidate universe?

**R2 vs R3 — Momentum Diversification**

Does a fixed 3/4/3 momentum-group allocation improve risk-adjusted performance relative to unrestricted Ridge selection?

**R2 vs R4 — Screening Design**

How does momentum-only candidate selection compare with momentum–volatility double sorting?

**B1 vs B2 — Portfolio Concentration**

Does holding 10 high-momentum stocks outperform a more diversified 30-stock momentum portfolio?

**All Strategies vs SPY and QQQ**

Do the strategies outperform investable market benchmarks after costs?

---

## 8. Portfolio Construction and Transaction Costs

The base transaction cost assumption is **20 basis points per traded side**.

Portfolios are rebalanced by trading the difference between current pre-trade weights and new target weights.

Stocks that remain selected are not automatically fully sold and repurchased. Excluded stocks are liquidated, while newly selected stocks are purchased.

R1 uses prediction-based long-only weighting, with a maximum individual stock allocation of 20%.

The remaining stock-selection strategies use equal weights.

All portfolio returns are evaluated after the assumed transaction costs.

The backtest is a historical research approximation, not a live execution simulator. Monthly stock P&L contributions should not be confused with complete realized trade-lifecycle accounting.

---

## 9. Evaluation Metrics

### Primary Portfolio Metric

**Net Annualized Return**

This measures annualized compounded portfolio performance after transaction costs.

### Additional Portfolio Metrics

- Annualized Volatility
- Sharpe Ratio
- Maximum Drawdown
- Turnover
- Transaction Costs
- Cumulative Wealth
- Annual and Subperiod Returns

### Prediction Metrics

- Training MSE and MAE
- Validation MSE and MAE
- Test MSE and MAE
- Out-of-Sample R²
- Rank Information Coefficient (Rank IC)

Portfolio returns and prediction accuracy are evaluated separately.

Sharpe Ratio is used for risk-adjusted performance comparison, not as the default Ridge training objective.

---

## 10. Installation and Execution

Create the environment:

```bash
conda create -n dsa5205-final python=3.11 -y
conda activate dsa5205-final
python -m pip install -r requirements.txt
```

Run the data pipeline:

```bash
python run_pipeline.py --frequency all
```

Run the main four-model experiments:

```bash
python run_models.py --frequency monthly --model ridge
python run_models.py --frequency monthly --model lasso
python run_models.py --frequency monthly --model rf
python run_models.py --frequency monthly --model xgb
```

The same commands can use `quarterly` or `semiannual`.

For the Ridge nine-strategy extension, open:

`notebooks/03_ridge_nine_strategy_comparison.ipynb`

Run the Notebook using the same Python environment and the existing processed market data.

The Ridge extension may use `strategies.py` at the project root.

---

## 11. Project Structure

```text
DSA5205_Project2/
├── config/
│   └── pipeline.json
├── src/
│   ├── pipeline.py
│   ├── models.py
│   └── backtest.py
├── notebooks/
│   ├── 01_data_pipeline.ipynb
│   ├── 02_unified_models.ipynb
│   └── 03_ridge_nine_strategy_comparison.ipynb
├── tests/
├── strategies.py
├── run_pipeline.py
├── run_models.py
├── requirements.txt
├── README.md
├── data/                  # Local, not uploaded to GitHub
└── outputs/               # Local, not uploaded to GitHub
```

The directory listing describes the intended project organization. The exact files available depend on the repository version.

---

## 12. GitHub Collaboration

The GitHub repository should contain:

- Python source code
- Jupyter Notebooks
- Configuration files
- Tests
- `requirements.txt`
- `README.md`

Generated data and experiment outputs should remain local and should not be committed to GitHub.

Recommended `.gitignore` entries:

```gitignore
data/
outputs/
__pycache__/
*.pyc
.ipynb_checkpoints/
.pytest_cache/
.DS_Store
.env
*.zip
```

Each team member can reproduce the dataset using the pipeline and run the assigned model with the same features, labels, and chronological split.

---

## 13. Project Requirements and Reporting

The final submission should follow Project2.pdf Sections 4.1–4.9.

The report should explain:

- Markets, assets, trading frequency, and execution timing
- Data acquisition and corporate-action adjustments
- Feature engineering and missing-value policy
- Momentum screening methodology
- Model selection and economic motivation
- Training, validation, and leakage prevention
- Portfolio construction and transaction costs
- Out-of-sample evaluation against benchmarks
- Model interpretation, limitations, and reproducibility

Required deliverables include a 15–25 page report, a 15–25 minute presentation video, and reproducible source code.

---

## 14. Limitations and Interpretation

The project uses historical prices of current S&P 500 constituents, which creates survivorship and constituent look-ahead bias. Historical data availability and Yahoo Finance corporate-action adjustments may introduce additional inaccuracies.

The original four-model experiment and the Ridge nine-strategy extension are distinct experimental settings. Their Ridge results should not be treated as identical when their training samples differ.

Several strategy extensions were designed after earlier test-period results had been inspected. These comparisons are therefore exploratory and should not be presented as untouched out-of-sample model selection.

The objective is to evaluate whether machine learning provides incremental predictive and economic value beyond traditional momentum strategies, rather than to guarantee superior investment performance.

**All results are for academic research and historical backtesting only.**