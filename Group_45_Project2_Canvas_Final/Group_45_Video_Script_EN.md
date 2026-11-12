# Group 45 · Project 2 video script

**One presenter · about 18 minutes, including the Notebook demonstration**

Rehearse the data build in a working copy using `REBUILD_DATA = True` in the Notebook setup cell. Prepare the market downloads in advance. Show the build and its output, then use the submitted monthly panels for the validation and backtest shown in the report. Italicised lines are screen cues.

## 00:00–01:15 · Research question and market

*Show the report cover and page 3.*

Hello. This is Group 45’s Project 2 on monthly equity selection. Our question is whether Ridge can improve on simple momentum when both methods choose from the same stocks.

Momentum has an economic motivation. When investors react gradually to new information, recent winners may continue to do well. We use Ridge to combine momentum with other information such as volatility, trend and liquidity, and ask whether that combination gives a better ranking.

We study US large-cap equities. We form a signal at month-end close, trade at the next session’s open and hold until the following rebalance open.

## 01:15–03:30 · Data, candidate screen and benchmarks

*Show the data build and the monthly data it produces. Open the benchmark table on page 4.*

Our inputs are daily prices and volumes from Yahoo Finance for a snapshot of current S&P 500 constituents. We apply price, liquidity and coverage filters. We then keep up to 300 eligible stocks ranked by their average dollar trading volume over the past 20 sessions.

We divide those stocks into high, middle and low momentum groups. Inside each group, we sort by 63-session realised volatility and take three stocks from the low-volatility group, four from the middle and three from the high group. That gives us thirty candidates each month, with representation across both momentum and volatility levels. We repeat the same screen at each decision date.

We use two momentum benchmarks. B1 buys the ten highest-momentum stocks from the broader liquid pool. B3 buys the ten highest-momentum stocks from the same thirty candidates used by Ridge. B3 is our direct comparison because it shares the candidate pool, equal weights, execution and cost rule. The ranking signal is what changes. SPY and QQQ are buy-and-hold benchmarks.

## 03:30–05:00 · Return target and features

*Show the target formula and feature table on page 5.*

The model predicts the holding-period return. We divide the adjusted open at the next rebalance by the adjusted open at entry, then subtract one. Adjusted prices account for splits and dividends. The execution date comes after the signal, and the return becomes known at the exit.

The model uses fifteen features available by the decision close. For example, twelve-minus-one momentum uses past returns while skipping the most recent month. Relative momentum compares a stock’s past return with SPY’s. Volatility and drawdown describe recent risk. Trading volume tells us how actively a stock trades, and the daily high-low range captures how far its price moved within the session.

For Ridge and Lasso, we standardise the inputs using the training rows. That puts the features on comparable scales before we penalise the coefficients. Random Forest and XGBoost use the original feature values. All four models use the same inputs and the same return target.

## 05:00–06:35 · Models and overfitting

*Show the Ridge objective and the parameter grid on page 6.*

Several features describe related price behaviour. For example, 63-session momentum and its market-relative version have a training correlation of 0.939. That motivates a model that can use correlated signals together.

Ridge adds an L2 penalty to squared prediction error. When features move together, a small change in the training sample can produce large changes in individual coefficients. Shrinkage reduces the influence of those large estimates. Lasso uses an L1 penalty, which can set some coefficients to zero. Random Forest and XGBoost let us check whether nonlinear effects help.

To control overfitting, we keep the parameter search small and limit the depth of the trees. We compare 23 settings in total. We fit them on the training block and score them on a later validation block.

## 06:35–08:10 · Chronological split

*Show the date blocks and eligible sample counts on page 7.*

We fixed the time blocks before fitting. We train on 2010 to 2019, validate on 2020 and 2021, and test on 2022 to 2025. Prices from 2009 supply the lookback for the first features in 2010.

A row enters training or validation only if its holding-period return has been realised by that block’s cutoff. For example, a November signal trades at the first December open and exits at the first January open. That label is still unavailable at the end of December.

The cutoff leaves 118 training months and 22 validation months. With thirty candidates each month, that gives 3,540 training observations and 660 validation observations. The two late 2019 months become eligible by the final cutoff. After model selection, we refit on all 4,260 eligible pre-test observations.

## 08:10–10:40 · Validation and choice of Ridge

*Run the Notebook validation cell. Show the MSE, Rank IC and Sharpe columns on page 8.*

For each model family, we choose the setting with the lowest validation mean squared error, or MSE. We also look at how well the model ranks the candidates. Rank IC is the correlation between the predicted and realised rankings each month, averaged across the validation months. Sharpe measures portfolio return per unit of volatility. Here we annualise it from monthly net returns and use a zero risk-free rate.

Lasso has the lowest MSE and the highest validation Sharpe among the four models. Its Sharpe is 0.926, compared with 0.918 for Ridge. Its selected fit sets six of the fifteen coefficients to zero. Ridge has the highest Rank IC, at 0.0438. The metrics reward different things, so the order depends on which column we look at.

We choose Ridge because the strategy acts on the ranking, and L2 shrinkage lets us combine related signals. Within Ridge, validation MSE selects an alpha of one thousand. We refit Ridge with that setting on the eligible pre-test observations and keep the fitted model fixed throughout the test.

## 10:40–13:00 · From forecasts to trades

*Show Figure 1 on page 9, then run the Ridge test cell in the Notebook.*

At each month-end, Ridge scores all thirty candidates. We buy the ten stocks with the highest scores and give each a target weight of ten percent. Figure 1 shows how the scores on 30 June 2025 become ten equal-weight positions.

Stocks earn different returns between rebalances, so their weights drift. At the next rebalance, we compare the drifted weights with the new targets. We charge 20 basis points on each amount bought or sold. Selling ten percent of the portfolio and buying another ten percent costs four basis points of portfolio value.

We apply the same cost rule to B1 and B3. The ETFs pay an entry fee and are then held over the same period. The test covers 48 monthly decisions, with holdings running from 1 February 2022 to 2 February 2026.

## 13:00–14:10 · Out-of-sample forecasts

*Show Figure 2 on page 10.*

We have thirty forecasts per month and 1,440 forecasts in total. The simple forecast benchmark gives every stock the average return observed in the eligible pre-test sample.

Ridge’s out-of-sample R-squared is 0.0187. At zero, the two forecasts would have the same squared error. Our positive value means Ridge’s squared error is about 1.87 percent lower than the historical mean forecast. This comparison uses all thirty candidates each month. The portfolio results measure what happens when we trade the ten highest scores.

## 14:10–15:55 · Portfolio and benchmark results

*Show the test table and Figure 3 on page 11.*

The cumulative net wealth chart starts at one dollar and compounds monthly returns after costs. Ridge finishes at about 4.19 dollars. Its net compound annual growth rate is 43.04 percent and its Sharpe is 1.272.

Ridge beats SPY and QQQ on both measures over this period, with higher volatility. B1 finishes highest in the cumulative chart and draws from the broader stock pool. B3 earns 51.51 percent per year net, with a Sharpe of 1.590. It also has lower turnover and a smaller drawdown at rebalance dates than Ridge.

For Ridge, monthly buys and sells average 126 percent of portfolio value. For B3, they average 94 percent. Costs take Ridge’s compound annual growth rate from 47.30 percent gross to 43.04 percent net. This shows why the course evaluates the strategy after trading costs.

## 15:55–16:55 · Performance by decision year

*Show the return and Sharpe panels in Figure 4 on page 12.*

Each year groups twelve monthly returns by the year of the signal. Ridge has a higher net return than B3 in 2023 and 2025. In 2023, B3 has the higher Sharpe. In 2025, Ridge leads on both measures.

Over the four years, B3 has the higher Sharpe in three. The widest return gap is in 2024, when B3 earns 122.68 percent and Ridge earns 70.77 percent. The yearly chart shows where the full-period advantage came from.

## 16:55–18:00 · Interpretation and reproducibility

*Show pages 13–14 and point to the run command in the README.*

Ridge improves the forecast relative to the historical mean. B3 delivers the stronger net portfolio in the same candidate pool. This is what we learned from separating prediction from trading performance.

Ridge’s beta to SPY is 1.64. In this sample, its returns respond more than one-for-one to moves in SPY. The stock pool is based on current constituents, which means it includes some firms before they entered the index and leaves out former members.

Our main result is the comparison with B3. We held the stock pool, weights and trading rules fixed, so we could see what changing the ranking signal did. The README and Notebook reproduce that comparison from the submitted data.
