# Report-Ready Sections

The numerical results below come from the executed monthly notebooks in this project. Returns are net of the assumed transaction costs, and Sharpe ratios use a zero risk-free rate.

## 4.4. Model Choice and Overfitting Control

### Economic story

The strategy is motivated by medium-term momentum: firms with relatively strong past performance may continue to perform well because information diffuses gradually and investors can underreact to persistent news. The candidate features describe several complementary aspects of this hypothesis, including returns over 21, 63, and 126 trading days, MOM2–12, momentum relative to SPY, moving-average trends, volatility, beta, drawdown, RSI, liquidity, volume, and price range. These variables are economically related rather than independent. For example, the momentum and moving-average measures contain overlapping information, while volatility, beta, and drawdown describe different aspects of risk.

This setting suggests a useful baseline in which several weak and correlated signals each contribute a modest amount of predictive information. Ridge regression matches this assumption. It retains all 15 features and uses an L2 penalty to shrink unstable coefficients rather than forcing a sparse subset to zero. Its approximately linear decision rule is transparent, and its coefficients can be inspected to determine whether the estimated relationships are economically plausible. Ridge does not directly solve a mean–variance portfolio optimization problem in this project; it produces return scores that subsequently determine stock rankings and, for R1, portfolio weights.

The other three models test meaningful alternatives to this story. LASSO assumes that a sparse subset of predictors is sufficient. Random Forest and XGBoost allow nonlinear effects and interactions, which may be relevant if momentum works only under particular volatility, trend, or liquidity conditions. Comparing all four algorithms therefore tests whether sparsity or nonlinear flexibility adds value beyond a shrunken linear model.

### Empirical model comparison and why Ridge is used for the extension

All four models use exactly the same original 30-stock candidate observations, features, target, chronological dates, and prediction-to-portfolio rule. Within each model family, hyperparameters are selected solely by minimum validation MSE over a small predeclared grid. The four best validation specifications are:

| Model | Best validation specification | Validation MSE | Validation R² |
|---|---|---:|---:|
| LASSO | alpha = 0.001 | 0.0222826 | 0.04501 |
| Random Forest | max_depth = 3, min_samples_leaf = 25 | 0.0224002 | 0.03997 |
| Ridge | alpha = 1,000 | 0.0224227 | 0.03901 |
| XGBoost | learning_rate = 0.03, max_depth = 2, 200 trees | 0.0227140 | 0.02652 |

LASSO has the lowest validation MSE. Ridge is therefore **not** presented as the empirically dominant algorithm or selected because of its test-set performance. Instead, we fix Ridge as the focal model for the strategy-design extension because its dense weak-signal assumption, shrinkage mechanism, stability under correlated features, and interpretability most directly match the economic question. Its validation MSE is only 0.63% above LASSO's and 0.10% above Random Forest's, so this modeling choice does not require ignoring a large validation-performance gap. The main experiment still reports all four test results, including cases in which another model performs better.

The main experiment and the Ridge extension answer different questions. The former isolates the prediction algorithm using the original 30-stock samples. The latter fixes the algorithm to Ridge and changes screening, selection, and allocation rules. The extension's common Ridge model is trained on a separate 60-stock momentum-only candidate sample; its results must not be treated as identical to main-experiment Ridge.

### Overfitting and leakage controls

Several controls limit overfitting and look-ahead risk:

1. **Time-respecting estimation.** Training uses 2010–2019, validation uses 2020–2021, and the untouched holdout evaluation covers 2022–2025. A training or validation observation is used only if its label has been fully realized by the relevant cutoff. No test observation is used to tune hyperparameters, refit coefficients, or choose the focal model.
2. **Small, fixed searches.** There are 6 Ridge candidates, 5 LASSO candidates, 4 Random Forest candidates, and 4 XGBoost candidates. The selection criterion is validation MSE, not test return or test Sharpe ratio.
3. **Shrinkage and scaling.** Ridge and LASSO standardize features inside pipelines fitted only on the permitted training observations. Ridge's selected alpha is 1,000 in both the main experiment and the separate extension. In the extension, validation MSE is 0.0218163 at alpha = 0.1 and improves to 0.0216392 at alpha = 1,000, supporting a meaningful amount of shrinkage.
4. **Tree complexity controls.** Random Forest uses shallow depth, minimum leaf sizes, feature subsampling, and 250 trees. XGBoost uses shallow trees, a small learning-rate/depth grid, row and column subsampling, an L2 penalty, and 200 trees.
5. **Fixed holdout rule.** The final model is refit once using eligible train and validation observations and remains static during testing. Features and portfolio holdings update at each rebalance, but test-period coefficients are not re-estimated.
6. **Consistent economic evaluation.** Every main-experiment model selects ten stocks at equal weights from the same candidates. Turnover is computed from drifted pre-trade weights, and returns include 20 basis points of cost per traded side. This prevents differences in portfolio rules from being mistaken for differences in prediction algorithms.

These controls do not eliminate all overfitting. In particular, the nine-strategy extension was designed after earlier test results had been inspected. It is accordingly reported as exploratory analysis rather than as an untouched confirmatory model-selection exercise.

## Interpretation

### What the four-model comparison shows

The holdout prediction results provide modest evidence of predictive information, but also show that model rankings depend on the metric:

| Model | Test MSE | Test MAE | Test R² | Mean monthly Rank IC |
|---|---:|---:|---:|---:|
| Ridge | 0.0150556 | 0.0827991 | 0.01874 | 0.06282 |
| LASSO | 0.0150908 | 0.0829610 | 0.01645 | 0.05956 |
| XGBoost | 0.0152794 | 0.0832055 | 0.00416 | 0.05414 |
| Random Forest | 0.0153073 | 0.0834109 | 0.00234 | 0.08957 |

Ridge has the lowest test MSE and MAE and the highest test R², although its R² of 1.87% is small in absolute terms. Random Forest has the highest average Rank IC despite having the highest MSE. This difference is economically meaningful: MSE evaluates the accuracy of return levels, whereas Rank IC evaluates whether stocks are ordered correctly within each month. A model can therefore rank candidates usefully without producing the most accurate numerical return forecasts.

The same distinction appears in portfolio performance:

| Model / reference | Net annualized return | Annualized volatility | Sharpe (rf = 0) | Maximum drawdown | Mean two-way turnover |
|---|---:|---:|---:|---:|---:|
| Random Forest Top 10 | 45.11% | 31.99% | 1.337 | −26.40% | 128.25% |
| Ridge Top 10 | 43.46% | 32.62% | 1.280 | −27.71% | 125.78% |
| LASSO Top 10 | 42.15% | 32.33% | 1.259 | −28.22% | 128.63% |
| XGBoost Top 10 | 36.62% | 30.89% | 1.174 | −28.42% | 132.50% |
| Momentum Top 10, same 30 candidates | 51.51% | 29.04% | 1.590 | −14.99% | 93.82% |
| SPY buy-and-hold | 12.72% | 15.27% | 0.863 | −19.67% | 2.08% |
| QQQ buy-and-hold | 14.81% | 19.93% | 0.796 | −25.56% | 2.08% |

Random Forest has the best ML portfolio return and Sharpe ratio even though Ridge has the best level-forecast metrics. None of the four ML Top-10 portfolios beats the simple momentum Top-10 baseline on return or Sharpe ratio, and all four have larger drawdowns and higher turnover. Thus, the main experiment does **not** support the claim that machine learning improves upon traditional momentum ranking within this candidate pool. The positive result is narrower: all four ML portfolios outperform SPY and QQQ over this particular holdout period, and the predictions contain some cross-sectional information. Given the data-universe limitations discussed below, these results should not be interpreted as estimates of implementable future alpha.

### What the Ridge extension shows

The separate 60-stock extension Ridge achieves test MSE of 0.0170500, MAE of 0.0918752, R² of −0.00680, and mean monthly Rank IC of 0.02180. The negative pooled R² means that its numerical return-level forecasts do not improve on the test-sample mean benchmark, while the small positive Rank IC indicates limited cross-sectional ordering information. These diagnostics are not directly comparable with the main Ridge statistics because the training and evaluation candidate samples differ.

The strategy comparisons give a mixed answer about the value of Ridge. R1 and R2 are extension results only: R1 changes the default weighting rule, while R2 changes both the candidate pool and the ranking rule. Neither is presented as evidence that the main default strategy succeeded.

- **Prediction-based weighting increased return but not risk-adjusted performance.** R1 holds exactly the same stocks as B1. R1 earns an 87.89% annualized return versus B1's 73.13%, a 14.76-percentage-point increase. However, R1's Sharpe ratio is slightly lower (1.642 versus 1.664), volatility is higher (44.93% versus 37.72%), maximum drawdown is worse (−21.47% versus −19.19%), and turnover is higher (80.32% versus 65.04%). Ridge-based allocation therefore concentrates risk more aggressively; it does not establish a risk-adjusted improvement.
- **Ridge ranking did not improve the original double-sorted pool.** R4 earns 44.21% annualized with a 1.304 Sharpe ratio, compared with B3's 51.51% and 1.590. R4 also has a larger maximum drawdown (−28.12% versus −14.99%) and higher mean turnover (127.39% versus 93.82%). Within the same 30-stock pool, replacing momentum ranks with the extension Ridge ranks is unsuccessful on these measures. This comparison also differs from main-experiment Ridge because R4's predictor is trained on the separate s60 sample.
- **Unrestricted selection beat the fixed momentum quota on return and Sharpe, but with a severe drawdown.** R2 earns 37.67% annualized with a 1.005 Sharpe ratio, whereas R3 earns 22.53% with a 0.778 Sharpe ratio. The 3/4/3 quota also increases turnover from 107.60% to 146.66%. R3's maximum drawdown is less severe (−38.99% versus −46.63%), but this does not offset its lower full-period return and Sharpe ratio.
- **Concentrated momentum was unusually strong.** B1 earns 73.13% annualized versus B2's 31.40%, and also has the higher Sharpe ratio (1.664 versus 1.193). This advantage comes with much higher volatility (37.72% versus 25.85%) and should not be generalized beyond the sample. It indicates that the strongest result comes primarily from concentrated momentum exposure rather than from machine learning.

The calendar-year results reinforce the need for caution. In 2022, R1 earns 8.42%, but R2, R3, and R4 lose 28.00%, 25.35%, and 13.03%, respectively; SPY and QQQ lose 8.82% and 18.79%. All nine strategies are positive in 2023–2025, and the largest strategy returns are concentrated in 2024 and 2025. The full-period rankings therefore depend strongly on the later years rather than representing a uniform benefit across market conditions.

Overall, the evidence supports three conclusions. First, historical price-based features contain modest predictive information. Second, prediction accuracy does not translate mechanically into superior portfolio performance. Third, the simple momentum baseline is difficult to improve: the most favorable Ridge result comes from changing weights on an already strong momentum portfolio, while Ridge stock selection and fixed momentum quotas often reduce performance.

### Limitations and what we learned

The most important limitation is that the stock universe consists of **current** S&P 500 constituents whose prices are backfilled historically. Delisted firms and historical index members that are no longer constituents are absent. This creates survivorship and constituent look-ahead bias and can materially inflate historical returns, especially for concentrated portfolios. The very large reported returns should consequently be read as properties of this research dataset, not as realistic investable performance estimates.

Additional limitations are the use of a static model throughout the four-year test period, a simplified 20-basis-point cost assumption without explicit bid–ask spread or market-impact modeling, only 48 monthly test observations, and zero-risk-free-rate Sharpe ratios. The extension involves several related comparisons after earlier test inspection, so no causal or confirmatory statistical claim is made. Future work should use a point-in-time universe including delisted stocks, pre-register a smaller set of strategy variants, and evaluate genuinely rolling or expanding out-of-sample windows on a new holdout period.

The main lesson is methodological rather than a claim of a guaranteed profitable strategy: clean timing, realized-label cutoffs, validation-only tuning, common portfolio rules, transaction costs, and simple investable benchmarks are essential. Weak or mixed economic results remain informative when the evaluation design is transparent.

## Reproducibility

The project requires Python 3.11. From the extracted project root, run:

```bash
conda create -n dsa5205-final python=3.11 -y
conda activate dsa5205-final
python -m pip install -r requirements.txt
python execute_notebooks.py --notebook all
```

The last command executes both notebooks in order. To run them separately:

```bash
python execute_notebooks.py --notebook data
python execute_notebooks.py --notebook models
```

`notebooks/01_data_pipeline.ipynb` obtains the lawful public data and creates all three frequency datasets. `notebooks/03_model_comparison_and_ridge_strategies.ipynb` contains the four-model comparison in Part I and the Ridge-only extension in Part II. The fixed seed is 42, and all paths are relative to the project root.

Key cross-references are:

- Main validation table: `outputs/four_model_comparison/monthly/validation_model_comparison.csv`.
- Main holdout prediction table: `outputs/four_model_comparison/monthly/test_prediction_comparison.csv`.
- Main net portfolio table: `outputs/four_model_comparison/monthly/test_portfolio_comparison.csv`.
- Main cumulative-return figure: `outputs/four_model_comparison/monthly/four_model_cumulative_wealth.png`.
- Ridge validation grid and curve: `outputs/ridge_nine_strategies/monthly/validation_grid.csv` and `ridge_validation_curve.png`.
- Nine-strategy performance and annual returns: `outputs/ridge_nine_strategies/monthly/strategy_summary.csv` and `annual_returns.csv`.
- Nine-strategy cumulative-return figure: `outputs/ridge_nine_strategies/monthly/nine_strategy_cumulative_wealth.png`.
- Ridge prediction diagnostics: `outputs/ridge_nine_strategies/monthly/prediction_metrics.json` and `ridge_rank_ic.png`.
- Trade and holdings audits: `outputs/ridge_nine_strategies/monthly/trades.csv` and `holdings.csv`.

A fresh data run requires internet access to Wikipedia and Yahoo Finance. Later runs reuse the local raw-data cache. Generated datasets and outputs are intentionally excluded from version control by `.gitignore`.
