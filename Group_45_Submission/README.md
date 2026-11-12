# DSA5205 Project 2 · Group 45

We fitted Ridge, LASSO, Random Forest and XGBoost on the 2010–2019 training data and compared them on the same 2020–2021 validation observations. Minimum validation MSE selects each family’s hyperparameters. We selected Ridge for its validation Rank IC and its L2 shrinkage of correlated features. Ridge was refitted at a penalty of 1,000 for the 48 test decisions in 2022–2025, with B1, B3, SPY and QQQ as fixed benchmarks.

## Reproduce the study

Use Python 3.12. On macOS with Homebrew, install the OpenMP runtime with `brew install libomp` if it is missing. From the project root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_primary_study.py
```

You can also open `notebooks/Project2_Final.ipynb` and run it with the project environment.

The run uses the included monthly model-ready panel and the liquid-universe panel for B1. A fresh vendor build is available through `python run_primary_study.py --rebuild-data`. That command needs access to Yahoo Finance and Wikipedia. It rebuilds the local panels from the vendor data available at the time of the run.

- `results/validation/validation_strategy_comparison.csv` contains the four models and four fixed benchmarks on the same validation dates. `validation_model_comparison.csv` contains the model-only comparison. `primary_decision.json` is written before the Ridge test and records the selected penalty and validation evidence.
- `results/oos/ridge_predictions.csv` contains 1,440 Ridge forecasts. `portfolio_metrics.csv`, `monthly_returns.csv`, `annual_returns.csv` and `annual_sharpe.csv` report Ridge and the fixed benchmarks.
- `figures/` contains the portfolio and prediction plots. `report/Group_45_Report.tex` and `report/generated/` contain the report source and its result tables.

The reference run used Python 3.12, numpy 1.26.4, pandas 2.2.3, pyarrow 18.1.0, scipy 1.17.1, scikit-learn 1.9.1 and xgboost 3.2.0. The four validation records and the five test portfolio metrics were checked against the pinned data and results.
