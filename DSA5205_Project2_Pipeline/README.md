# DSA5205 Project 2 — Reproducible Equity Screening and ML Trading Pipeline

**Version:** final research implementation (2026-11-05). **Purpose:** historical research/backtesting only; no live orders.

## 0. Important disclosure (read before interpreting results)

The program **automatically downloads the latest/current Wikipedia S&P 500 member list** and retrieves each available ticker's historical Yahoo Finance OHLCV using `yfinance`. **It does not reconstruct historical S&P 500 membership.** Therefore historical backtests suffer from **survivorship / constituent look-ahead bias**, even though prices are genuine. A ticker is not assumed to have traded before its IPO: eligibility is assessed independently each decision date. Delisted former members not on today's list are absent; downloaded current members may not have been index members historically. The resulting historical sample must be described as *current constituents backfilled through time*, **not** a point-in-time historical S&P 500 universe. No amount of momentum screening removes this bias. Do not claim an unbiased investable strategy or guaranteed performance.

Yahoo data can be missing, revised or subject to corporate-action errors. Full data quality cannot be guaranteed without independent verification. Downloads require internet access. The provided code does **not** include predownloaded data. Failed ticker downloads are retried; small numbers are excluded and documented in `data/raw/download_audit.json` and `download_status.csv`. An >10% failure rate stops the build. Stocks that IPO late are *not* excluded for lacking early history.

## 1. Research idea and economic motivation (§4.1–4.4)

A monthly/quarterly/semiannual US equity cross-sectional strategy. Historical MOM2–12 captures medium-term momentum while 63-day realized volatility represents risk. From the eligible current-constituent sample, rank by ADV20 and retain up to 300 stocks. Divide by momentum tercile (High/Medium/Low), then within each by volatility tercile (Low/Medium/High). Select 3/4/3 stocks per momentum tercile, with High selecting highest momentum, Low selecting lowest, and Medium selecting those nearest the cell median; ticker breaks ties. This yields 30 stocks on each valid decision date. If a cell lacks capacity, log the failure; never backfill with a future winner.

The four pooled ML regressors (Ridge, LASSO, Random Forest, XGBoost) use identical 15 *past-only* numeric features to predict the **next holding-period simple total-return rate**. `target_forward_return=0.03` means **+3%**, not $3 or an absolute value. The model picks predicted Top 10 equally weighted (10% each). Compare to momentum-only Top 10, equal-weight 30, and buy-and-hold SPY and QQQ. Ridge/LASSO control estimation variance by L2/L1 shrinkage; RF/XGB control overfitting with restricted depth, minimum leaf size, subsampling, and fixed small hyperparameter grids. Nonlinear methods can capture feature interactions; this is a hypothesis, not a guarantee.

## 2. Timing (§4.1, §4.2, §4.5)

- Market: US equities, USD, US exchange sessions; dates in Yahoo daily US market calendar.
- Raw price download: 2009-01-01 to 2026-11-01 **exclusive** (settlement buffer for 2025 H2 semiannual labels). Research decision dates: 2010-01-01 through 2025-12-31.
- Primary experiment: monthly. Extensions: quarterly (calendar quarter-end) and semiannual (June/December end).
- Signal: final available SPY trading-session close in each period. Orders assumed filled **next trading-session open**. Exit at the next scheduled rebalance's **open**. No same-close execution.
- `target_forward_return = adjusted_open(exit) / adjusted_open(entry) - 1`; `adjusted_open = raw_open * adj_close/raw_close`. The adjusted-open approximation incorporates Yahoo dividend/split adjustments and must be independently checked for representative events.
- `spy_return` and `qqq_return` use identical execution/exit dates. SPY/QQQ-relative labels are auxiliary only, not training targets.
- Split by **decision date**: Train 2010–2020, Validation 2021–2022, Test 2023–2025. Candidate training labels must also have **label_end_date <= 2020-12-31**; validation labels <= 2022-12-31. Final model refits Train+Validation labels realized by 2022-12-31. Test labels are never used in fitting or tuning.
- **Important:** Test periods for quarterly/semiannual may extend into 2026. A row belongs to the 2025 Test split by *decision date*, while the holding period can end in 2026. Disclose this explicitly.

## 3. Feature definitions (§4.2)

| Feature | Formula / lookback |
|---|---|
| `mom_21`, `mom_63`, `mom_126` | AdjClose(t)/AdjClose(t-L)-1, L=21/63/126 trading sessions |
| `mom_2_12` | Previous calendar month-end AdjClose / 12-months-ago month-end AdjClose - 1 |
| `relative_mom_63` | Stock 63-session momentum minus SPY 63-session momentum (past only) |
| `vol_21`, `vol_63` | Std. of daily adjusted close returns × sqrt(252), 21/63 sessions |
| `beta_252` | Rolling covariance(stock, SPY)/variance(SPY), 252 paired daily returns |
| `drawdown_63` | AdjClose / trailing 63-session maximum - 1 |
| `ma_ratio_50`, `ma_ratio_200` | AdjClose / 50/200-session moving average - 1 |
| `rsi_14` | 14-session simple-average RSI, 0–100 |
| `ln_adv_20` | log(1 + mean(raw close × volume, 20 sessions)) |
| `volume_ratio_21` | Volume / 21-session mean volume |
| `range_hl` | (raw high - raw low) / raw close |

## 4. Cleaning, NaN, IPO, quality (§4.2, Appendix B)

Three layers: `data/raw/market_data.parquet` (vendor observations, unchanged); `data/processed_with_nan/full_feature_panel.parquet` (daily calendar-aligned features, quality flags, missing values retained); `data/processed_ready/{frequency}/model_ready.parquet` (valid selected rows, numeric X and Y finite). `screened_with_label_status.parquet` retains the **original selected** stocks even when their future label cannot be verified. Never reselect based on future label availability.

- No global raw-price imputation. Each stock is aligned to SPY's session calendar so missing days are counted, not silently skipped.
- Rolling statistics permit approximately 5% missing observations; screening additionally requires >=95% observed raw daily OHLCV over 21, 63, and 252 sessions. `mom_2_12` needs valid month-end endpoints. Core screening factors (`mom_2_12`, `vol_63`, `adv_20`) are **never feature-ffilled**.
- Other computed ML features may be forward-filled for at most two sessions, only within a ticker; raw coverage flags remain unchanged. No labels or execution prices are imputed.
- Stocks with insufficient history (including recent IPOs) simply become eligible later; no complete 2009–2026 history requirement.
- Invalid/missing execution prices produce missing labels. Such cases are reported and **not silently replaced with another stock**. A model-ready file may contain fewer than 30 labeled rows for that date; evaluate coverage and do not present that period as a complete 30-stock strategy without qualification.
- Missing delisting payoff, trading halts and absent historical former members cannot be resolved by Yahoo alone. This is a material backtest limitation.

## 5. Installation and exact commands (§6–7)

From the **unzipped project root** (not the notebooks directory):

```bash
conda create -n dsa5205-final python=3.11 -y
conda activate dsa5205-final
python -m pip install -r requirements.txt
python -m ipykernel install --user --name dsa5205-final --display-name "Python (DSA5205 Final)"
python run_pipeline.py --frequency all
python run_models.py --frequency monthly --model ridge
python run_models.py --frequency monthly --model lasso
python run_models.py --frequency monthly --model rf
python run_models.py --frequency monthly --model xgb
python -m pytest -q
```

After a successful download, rebuild without network calls using `python run_pipeline.py --use-cache --frequency all`. To run all three frequencies for any model, invoke `run_models.py` separately with `--frequency monthly`, `quarterly`, `semiannual`.

Notebooks: `notebooks/01_data_pipeline.ipynb` runs the pipeline and charts; `notebooks/02_unified_models.ipynb` runs a selected model. Run them with the project root as working directory or use the explicit root locator in the notebook. No hard-coded Mac user paths. Data download may take considerable time and be rate limited by Yahoo.

## 6. Expected outputs and team handoff

```
data/raw/market_data.parquet
 data/raw/download_status.csv
 data/raw/download_audit.json
 data/raw/ticker_universe.json
 data/processed_with_nan/full_feature_panel.parquet
 data/processed_ready/{monthly,quarterly,semiannual}/model_ready.parquet
 data/processed_ready/{frequency}/screened_with_label_status.parquet
 data/processed_ready/{frequency}/feature_columns.json
 data/processed_ready/{frequency}/model_contract.json
 outputs/screening_audit_{frequency}.csv
 outputs/quality_summary_{frequency}.json
 outputs/figures/{frequency}/*.png
 outputs/models/{frequency}/{ridge,lasso,rf,xgb}/training_validation_grid.csv
 outputs/models/{frequency}/{model}/test_predictions.csv
 outputs/models/{frequency}/{model}/portfolio_returns.csv
 outputs/models/{frequency}/{model}/rebalance_trade_log.csv
 outputs/models/{frequency}/{model}/monthly_stock_pnl.csv
 outputs/models/{frequency}/{model}/performance_summary.csv
```

Model handoff: **all models read the same `model_ready.parquet`, `feature_columns.json`, and `target_forward_return`**. Do not use target, future benchmark returns, group labels, future execution prices, or `split` as predictive X. Ridge/LASSO fit StandardScaler inside the training pipeline only; RF/XGB need not scale. `run_model()` logs train/validation MSE, MAE and R2 for each candidate, chooses lowest validation MSE, refits with realized Train+Validation labels, and predicts the untouched Test. Tree models are not guaranteed to outperform. All four share seed 42 and the same selected candidate rows.

## 7. Portfolio accounting and limitations (§4.8–4.9)

Backtest buys each month's predicted Top 10 at the next open, target weight 10% each, then measures the next holding-period adjusted-open return. At the next rebalance, prior weights drift with realized returns, and only **the difference** from new targets is traded; exits are fully sold. Two-way traded notional = sum(abs(target_weight - pretrade_weight)); base cost = 0.002 × that turnover (20 bps **per side**). SPY/QQQ are buy-and-hold comparators with a one-time initial entry cost. Weight-level `monthly_stock_pnl.csv` contains per-stock period P&L contribution, **not** FIFO/average-cost realized dollar profit across a multi-month holding. There is no live order routing, execution market impact model, tax, borrowing, or shorting.

For rigorous results, inspect `screened_with_label_status.parquet`: if missing stock labels or failed screens exist in Test, do not treat the remaining subset as a full-investment performance estimate. Check coverage, report exclusion and run robustness. The portfolio engine assumes a complete, executable candidate set at each evaluated date.

## 8. Report and video checklist (Project2.pdf)

§4.1 assets, timing, benchmarks, sources; §4.2 label equation, feature table, missing policy, split/dividend check; §4.3 SPY/QQQ/Momentum benchmarks; §4.4 economic rationale and complexity control for all four models; §4.5 chronological split and label availability; §4.6 solver/training, versions and seed; §4.7 small predefined grid and validation rule; §4.8 score-to-weight and turnover/cost formula; §4.9 OOS net annualized return, volatility, Sharpe, cumulative wealth vs SPY/QQQ, OOS R2, and subperiod return/Sharpe. Interpret weak or strong OOS results without optimizing on the test set. Include limitations and data coverage tables.

Deliver **15–25 page PDF report (25 points)**, **15–25 minute video (10 points)**, code ZIP or GitHub, exact README commands, package versions, contribution note and reproducible figures. The PDF explicitly says **test only once at the end**. Since earlier 2023–2025 Ridge results have already been inspected, any new model choices based on them must be labeled exploratory, not an untouched holdout. No real trading.

## 9. Research references and data sources

- Course specification: `Project2.pdf`, §§4.1–4.9, Appendix B.
- Yahoo Finance through yfinance: https://ranaroussi.github.io/yfinance/
- Current S&P 500 constituents: https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
- Kelly, Malamud, Zhou (2024), *The Virtue of Complexity in Return Prediction*.
- Asness, Moskowitz, Pedersen (2013), *Value and Momentum Everywhere*.

## 10. Known caveats / future improvements

This package cannot certify a bias-free historical universe without point-in-time membership and delisting prices. Current-constituent backfills can substantially inflate performance. For stronger claims use CRSP / point-in-time constituents and actual delisting payoffs. `adjusted_open` is a vendor-based total-return approximation, not an independently observed total-return open. All reported results should be qualified accordingly. The downloaded sample is not a source of historical S&P 500 membership truth.

### Extra fail-closed safeguards

`run_models.py` **refuses** to backtest if any Test decision date has fewer than 30 labeled candidates. The user must investigate missing executions/labels, rather than silently backtesting only surviving winners. `run_pipeline.py` exports `outputs/qa_{frequency}.json` and `outputs/corporate_action_review.csv`; inspect corporate-action flags manually, including at least one dividend and one split as requested in Appendix B. Passing automated QA is not equivalent to proving adjusted Yahoo prices are correct.

`run_models.py` also exports `yearly_return_sharpe.csv`, `yearly_annualized_return.png`, `yearly_sharpe_rf0.png`, `cost_sensitivity.csv` (5/10/20/40 bps), `monthly_rank_ic.csv`, `cumulative_wealth.png`, and `test_oos_r2.txt`. The last R2 uses the **test sample mean** as the descriptive denominator; a historical-mean forecast baseline would require separate evaluation and is optional under §4.3.
