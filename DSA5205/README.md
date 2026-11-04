# DSA5205 Project 2 — Daily Market Timing on SPY with Ridge Shrinkage

**Group_X** (replace with your group number and add the team roster on the report title
page before submitting).

A daily market-timing strategy on SPY: 12 technical features after each close → ridge
regression (shrinkage penalty selected on a validation window) → volatility-managed
position executed at the next day's open, evaluated out of sample on 2020–2026 with
excess-return (cash/financing) accounting, net of a 1 bp transaction cost. Benchmarks:
buy-and-hold SPY, an AR(1) forecast baseline, a mean-forecast + volatility-management
benchmark, a fixed-exposure diagnostic, and CAPM attribution.

## Layout

```
DSA5205/
├── README.md                # this file
├── requirements.txt         # pinned environment
├── report.md                # report source (markdown)
├── Group_X_Report.pdf       # the report (build artifact)
├── code/
│   ├── fetch_data.py        # download SPY (Yahoo) + T-bill rate (FRED/Yahoo) -> data/
│   ├── features.py          # feature/label construction (imported by run_backtest.py)
│   ├── run_backtest.py      # train/validate/test, strategies, metrics -> output/results/
│   ├── run_robustness.py    # Appendix D exploratory diagnostics -> robustness.csv
│   ├── make_figures.py      # figures 1-4 -> output/figures/
│   ├── build_report.py      # report.md -> Group_X_Report.pdf (needs pandoc)
│   └── test_backtest.py     # boundary / drawdown / naive-forecast assertions
├── data/                    # committed cache: spy_daily.csv, dtb3.csv
└── output/
    ├── results/             # metrics.csv, capm.csv, lambda_selection.csv, oos_*.csv,
    │                        # robustness.csv, yearly_stats.csv, run_config.json
    └── figures/             # fig1_oos_r2.png .. fig4_lambda_selection.png
```

## Setup

```
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Building the PDF additionally requires the `pandoc` binary on the PATH (a system
dependency, only needed for `build_report.py`; the analysis itself needs nothing beyond
`requirements.txt`).

## Reproduce from the committed cache (no network needed)

The `data/` folder is committed, so the full pipeline runs offline. From `DSA5205/`:

```
python code/test_backtest.py     # sanity assertions (boundary labels, drawdown, naive)
python code/run_backtest.py      # -> output/results/*.csv, prints metrics + CAPM tables
python code/run_robustness.py    # -> output/results/robustness.csv (Appendix D)
python code/make_figures.py      # -> output/figures/*.png + yearly_stats.csv
python code/build_report.py      # optional: rebuild Group_X_Report.pdf (needs pandoc)
```

## Refresh the data (optional, needs network)

```
python code/fetch_data.py
```

- The analysis window is pinned (`START`/`END` in `fetch_data.py`, end 2026-10-30) so
  results do not drift as new data arrives.
- Cache safety: both datasets are downloaded and validated (non-empty, required columns,
  no duplicate dates, no non-positive prices, correct end date) **before** any cache file
  is replaced, and writes are atomic (temp file + rename). A failed or empty download
  aborts with a clear error and never corrupts the committed cache.
- Risk-free leg: FRED `DTB3` is tried first; if the endpoint is unreachable the script
  falls back to Yahoo `^IRX` (same 3-month T-bill instrument).
- The script also verifies adjustment continuity on the two most recent SPY dividend
  events (manual total return vs vendor adjusted return).

## Expected outputs

- `output/results/metrics.csv` — post-cost annualized return/volatility/Sharpe, max
  drawdown, turnover, OOS R² for ridge (validation + test), mean-forecast, AR(1),
  fixed-exposure (exploratory), and buy-and-hold (test).
- `output/results/capm.csv` — CAPM alpha/beta with Newey–West t-statistics.
- `output/results/lambda_selection.csv` — validation Sharpe/return/MSE/OOS-R² for each of
  the 10 ridge penalties in the pre-set grid (selected: λ = 30,000).
- `output/results/robustness.csv` — Appendix D diagnostics (cost 0.5/1/2 bp, cap ±1.0,
  long-only), computed from the frozen configuration, never used for re-selection.
- `output/results/oos_predictions.csv`, `oos_strategy_returns.csv` — day-by-day test
  forecasts, positions, and net returns.
- `output/results/run_config.json` — seed (42), grid, selected λ, fitted coefficients,
  γ values, package versions.
- `output/figures/fig1..fig4 .png` and `Group_X_Report.pdf` (18 pages).

Headline test results (2020-01-02 → 2026-10-28, 1,715 days, net of costs): ridge
strategy Sharpe ≈ 0.64 (ann. return ≈ 10.5%, vol ≈ 12.3%, max drawdown ≈ −14.1%),
mean-forecast + vol mgmt ≈ 0.65, AR(1) baseline ≈ 0.42, buy-and-hold ≈ 0.69
(vol ≈ 19.1%, drawdown ≈ −32.1%). CAPM alpha ≈ +1.8%/yr, Newey–West t ≈ 0.59
(not significant) — attribution shows the risk profile comes mainly from lower
market exposure and volatility-managed sizing, not from the 12-feature forecast.

## Reproducibility notes

- Fixed seed 42 (the pipeline is deterministic — ridge is solved in closed form; the
  seed is set defensively in `run_backtest.py`).
- Relative paths only; no hard-coded local directories; no API keys or secrets.
- Timing/leakage design: features use data up to the decision close only; execution is
  at the *next day's open*; the label is the open-to-open excess return over the holding
  day and is only treated as realized at the open of t+2 — every fit/selection sample
  excludes boundary rows whose label is unrealized at the first out-of-sample decision
  (assertions in `split_dataset`, tested in `test_backtest.py`); the standardizer and
  risk scaler are fitted on training data and applied forward; λ is chosen on 2015–2019
  validation; the frozen configuration receives one confirmatory evaluation on
  2020–2026, with post-hoc diagnostics clearly labelled exploratory.

## Video checklist (what to show)

1. `python code/fetch_data.py` — data build, adjustment + dividend-event checks (or the
   committed cache), and the cache-safety behavior.
2. `python code/run_backtest.py` — the λ-selection table, the metrics table, CAPM output.
3. `python code/make_figures.py` — Figures 1–4.
4. Walk through the economic story (many weak collinear signals → ridge shrinkage as
   bias–variance control), the overfitting controls (validation-only λ, frozen test,
   caps), the score→position rule π_t = clip(ŷ_t/(γ·σ̂²_t), ±1.5) executed at the next
   open with 1 bp costs, and the out-of-sample results including attribution — what
   worked (shrinkage, vol-managed sizing, cost awareness, executable timing) and what did
   not (no statistically significant alpha; features add little over vol management).
