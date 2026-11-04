"""Train, validate, and test the ridge market-timing strategy.

Timing and accounting (see features.py docstring):
  - Signal after close t, execution at open t+1, held to open t+2.
  - Models predict the EXCESS open-to-open return (fwd_excess).
  - Portfolio return: r_p = r_f + pi * fwd_excess - COST * |pi_t - pi_{t-1}|.
    Uninvested capital earns r_f; leverage borrows at r_f; short proceeds earn
    r_f (SPY borrow fees assumed negligible).

Design (leakage control, brief section 4.5):
  - Train:       2005-01-01 .. 2014-12-31  (fit scaler, ridge, AR(1); calibrate gamma)
  - Validation:  2015-01-01 .. 2019-12-31  (choose lambda from a small pre-set grid
                 by validation strategy Sharpe, single forward split)
  - Test:        2020-01-01 .. last date   (confirmatory evaluation of the frozen
                 configuration; exploratory diagnostics are computed afterwards and
                 labelled as such -- they were not used to re-select anything)
  - Boundary rule: the label of decision day t is realized only at the open of
    day t+2, so every fit/selection sample excludes rows whose label_end_date
    is after the first out-of-sample decision date (one boundary row per split;
    enforced by assertions in split_dataset).

Attribution benchmarks (brief section 4.3 + attribution analysis):
  - buy-and-hold SPY;
  - AR(1) forecast through the identical position rule;
  - mean-forecast: the constant training-mean excess return through the
    identical volatility-managed position rule (isolates the incremental value
    of the 12-feature ridge forecast);
  - fixed-exposure (EXPLORATORY): a constant SPY weight equal to the ridge
    strategy's average test exposure (checks whether performance simply
    reflects lower market exposure);
  - CAPM regression of the strategy's net excess return on the market excess
    return with Newey-West t-statistics.

Outputs (under output/results/):
  lambda_selection.csv, metrics.csv, oos_predictions.csv,
  oos_strategy_returns.csv, yearly_stats.csv, capm.csv, run_config.json
"""

import json
import os
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression, Ridge

from features import FEATURES, build_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "output", "results")

TRAIN_END = "2014-12-31"
VAL_END = "2019-12-31"
LAMBDA_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0, 300000.0]
PI_MAX = 1.5
COST = 1e-4          # 1 bp per unit turnover
SEED = 42
TRADING_DAYS = 252
NW_LAG = 5           # Newey-West lag for the CAPM regression


def standardize(fit_df: pd.DataFrame, *dfs: pd.DataFrame):
    """Z-score using mean/std fitted on fit_df only, applied forward."""
    mu = fit_df[FEATURES].mean()
    sd = fit_df[FEATURES].std().replace(0.0, 1.0)
    return tuple((d[FEATURES] - mu) / sd for d in dfs)


def positions(yhat: np.ndarray, sigma2: np.ndarray, gamma: float) -> np.ndarray:
    return np.clip(yhat / (gamma * sigma2), -PI_MAX, PI_MAX)


def calibrate_gamma(yhat: np.ndarray, sigma2: np.ndarray) -> float:
    """Calibrate gamma so the average absolute PRE-CLIP raw position
    |yhat / (gamma * sigma2)| equals 1 on the fitting sample; the +/-PI_MAX
    cap is applied afterwards (so realized mean|pi| can be below 1)."""
    return float(np.abs(yhat / sigma2).mean())


def strategy_returns(pi: np.ndarray, fwd_excess: np.ndarray, rf: np.ndarray):
    """Net-of-cost portfolio returns r_f + pi*excess - cost*turnover, and turnover."""
    pi_prev = np.concatenate([[0.0], pi[:-1]])
    turnover = np.abs(pi - pi_prev)
    return rf + pi * fwd_excess - COST * turnover, turnover


def max_drawdown(ret: np.ndarray) -> float:
    """Max drawdown relative to the initial capital of 1 (a first-day loss counts)."""
    wealth = np.concatenate([[1.0], np.cumprod(1.0 + ret)])
    peak = np.maximum.accumulate(wealth)
    return float((wealth / peak - 1.0).min())


def perf(ret: np.ndarray, rf: np.ndarray, turnover: np.ndarray | None = None) -> dict:
    n = len(ret)
    excess = ret - rf
    out = {
        "n_days": n,
        "ann_return": float(np.prod(1.0 + ret) ** (TRADING_DAYS / n) - 1.0),
        "ann_vol": float(ret.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "sharpe": float(excess.mean() / ret.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "max_drawdown": max_drawdown(ret),
    }
    if turnover is not None:
        out["avg_daily_turnover"] = float(turnover.mean())
        out["ann_cost_drag"] = float(COST * turnover.mean() * TRADING_DAYS)
    return out


def oos_r2(y_true: np.ndarray, y_hat: np.ndarray, y_naive: np.ndarray) -> float:
    """Out-of-sample R^2 vs the recursive historical-mean naive forecast."""
    sse = ((y_true - y_hat) ** 2).sum()
    sst = ((y_true - y_naive) ** 2).sum()
    return float(1.0 - sse / sst)


def capm_regression(y_excess: np.ndarray, mkt_excess: np.ndarray) -> dict:
    """OLS of strategy excess on market excess with Newey-West t-stats."""
    T = len(y_excess)
    X = np.column_stack([np.ones(T), mkt_excess])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y_excess
    e = y_excess - X @ beta
    # Newey-West covariance
    S = np.zeros((2, 2))
    for t in range(T):
        xt = X[t][:, None]
        S += (e[t] ** 2) * (xt @ xt.T)
    for lag in range(1, NW_LAG + 1):
        w = 1.0 - lag / (NW_LAG + 1)
        for t in range(lag, T):
            xt, xl = X[t][:, None], X[t - lag][:, None]
            S += w * e[t] * e[t - lag] * (xt @ xl.T + xl @ xt.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    return {
        "alpha_daily": float(beta[0]),
        "alpha_ann": float(beta[0] * TRADING_DAYS),
        "beta": float(beta[1]),
        "t_alpha": float(beta[0] / se[0]),
        "t_beta": float(beta[1] / se[1]),
        "nw_lag": NW_LAG,
    }


def split_dataset(ds: pd.DataFrame):
    """Chronological train/validation/test split with label-availability control.

    The label of decision day t is only realized at the open of day t+2, so a
    fitting (or model-selection) sample may only contain rows whose label was
    already realized at the first out-of-sample decision time. This drops the
    last decision row at each boundary and is enforced by assertions.
    """
    val_start = ds.loc[ds["date"] > TRAIN_END, "date"].min()
    test_start = ds.loc[ds["date"] > VAL_END, "date"].min()

    train = ds[(ds["date"] <= TRAIN_END)
               & (ds["label_end_date"] <= val_start)].reset_index(drop=True)
    val = ds[(ds["date"] > TRAIN_END) & (ds["date"] <= VAL_END)
             & (ds["label_end_date"] <= test_start)].reset_index(drop=True)
    test = ds[ds["date"] > VAL_END].reset_index(drop=True)
    trainval = ds[(ds["date"] <= VAL_END)
                  & (ds["label_end_date"] <= test_start)].reset_index(drop=True)

    # no fit/selection sample may contain a label unrealized at the first OOS decision
    assert train["label_end_date"].max() <= val_start, "unrealized label in train"
    assert val["label_end_date"].max() <= test_start, "unrealized label in validation"
    assert trainval["label_end_date"].max() <= test_start, "unrealized label in trainval"
    return train, val, test, trainval


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    ds = build_dataset().dropna(subset=["naive_excess"]).reset_index(drop=True)
    train, val, test, trainval = split_dataset(ds)
    print(f"train {len(train)}  val {len(val)}  test {len(test)} "
          f"({train['date'].iloc[0].date()} .. {test['date'].iloc[-1].date()})")
    print(f"[boundary] train labels realized by {ds.loc[ds['date'] > TRAIN_END, 'date'].min().date()}; "
          f"val/trainval labels realized by {ds.loc[ds['date'] > VAL_END, 'date'].min().date()}")

    target = "fwd_excess"

    # ---------------- validation: choose lambda ----------------
    Xtr_s, Xval_s = standardize(train, train, val)

    rows, val_cache = [], {}
    for lam in LAMBDA_GRID:
        m = Ridge(alpha=lam).fit(Xtr_s, train[target])
        yhat_val = m.predict(Xval_s)
        gamma = calibrate_gamma(m.predict(Xtr_s), train["sigma2_21"].to_numpy())
        pi = positions(yhat_val, val["sigma2_21"].to_numpy(), gamma)
        net, _to = strategy_returns(pi, val[target].to_numpy(), val["rf_next"].to_numpy())
        p = perf(net, val["rf_next"].to_numpy())
        rows.append({
            "lambda": lam, "val_sharpe": p["sharpe"], "val_ann_return": p["ann_return"],
            "val_mse": float(((val[target] - yhat_val) ** 2).mean()),
            "val_oos_r2": oos_r2(val[target].to_numpy(), yhat_val, val["naive_excess"].to_numpy()),
        })
        val_cache[lam] = (m, gamma)
    lam_sel = pd.DataFrame(rows)
    lam_sel.to_csv(os.path.join(RESULTS, "lambda_selection.csv"), index=False)
    best_lam = float(lam_sel.loc[lam_sel["val_sharpe"].idxmax(), "lambda"])
    print(f"[select] lambda grid {LAMBDA_GRID} -> best lambda = {best_lam} (by validation Sharpe)")
    print(lam_sel.round(4).to_string(index=False))

    # ---------------- test: refit on train+val, confirmatory evaluation ----------------
    Xtv_s, Xte_s = standardize(trainval, trainval, test)

    ridge = Ridge(alpha=best_lam).fit(Xtv_s, trainval[target])
    yhat_ridge = ridge.predict(Xte_s)
    gamma_ridge = calibrate_gamma(ridge.predict(Xtv_s), trainval["sigma2_21"].to_numpy())

    ar1 = LinearRegression().fit(Xtv_s[["ret_1"]], trainval[target])
    yhat_ar1 = ar1.predict(Xte_s[["ret_1"]])
    gamma_ar1 = calibrate_gamma(ar1.predict(Xtv_s[["ret_1"]]), trainval["sigma2_21"].to_numpy())

    # mean-forecast benchmark: constant training-mean excess, same position rule
    yhat_mean = float(trainval[target].mean())
    gamma_mean = calibrate_gamma(np.full(len(trainval), yhat_mean), trainval["sigma2_21"].to_numpy())

    sig2 = test["sigma2_21"].to_numpy()
    exs = test[target].to_numpy()
    fwd = test["fwd_ret"].to_numpy()
    rf = test["rf_next"].to_numpy()
    naive_te = test["naive_excess"].to_numpy()

    pi_ridge = positions(yhat_ridge, sig2, gamma_ridge)
    pi_ar1 = positions(yhat_ar1, sig2, gamma_ar1)
    pi_mean = positions(np.full(len(test), yhat_mean), sig2, gamma_mean)
    net_ridge, to_ridge = strategy_returns(pi_ridge, exs, rf)
    net_ar1, to_ar1 = strategy_returns(pi_ar1, exs, rf)
    net_mean, to_mean = strategy_returns(pi_mean, exs, rf)

    # exploratory diagnostic: fixed exposure equal to the ridge strategy's
    # average test exposure (uses test information -> exploratory only)
    pi_fixed = np.full(len(test), pi_ridge.mean())
    net_fixed, to_fixed = strategy_returns(pi_fixed, exs, rf)

    # validation-period performance of the final-chosen lambda (for the report)
    m_val, g_val = val_cache[best_lam]
    yhat_v = m_val.predict(Xval_s)
    pi_v = positions(yhat_v, val["sigma2_21"].to_numpy(), g_val)
    net_v, to_v = strategy_returns(pi_v, val[target].to_numpy(), val["rf_next"].to_numpy())

    metrics = pd.DataFrame([
        {"model": "ridge", "split": "validation", **perf(net_v, val["rf_next"].to_numpy(), to_v),
         "oos_r2": oos_r2(val[target].to_numpy(), yhat_v, val["naive_excess"].to_numpy())},
        {"model": "ridge", "split": "test", **perf(net_ridge, rf, to_ridge),
         "oos_r2": oos_r2(exs, yhat_ridge, naive_te)},
        {"model": "ar1_baseline", "split": "test", **perf(net_ar1, rf, to_ar1),
         "oos_r2": oos_r2(exs, yhat_ar1, naive_te)},
        {"model": "mean_forecast", "split": "test", **perf(net_mean, rf, to_mean),
         "oos_r2": oos_r2(exs, np.full(len(test), yhat_mean), naive_te)},
        {"model": "fixed_exposure_exploratory", "split": "test", **perf(net_fixed, rf, to_fixed),
         "oos_r2": np.nan},
        {"model": "buy_and_hold", "split": "test", **perf(fwd, rf), "oos_r2": np.nan},
    ])
    metrics.to_csv(os.path.join(RESULTS, "metrics.csv"), index=False)

    # ---------------- attribution: CAPM on net strategy excess ----------------
    capm = pd.DataFrame([
        {"model": "ridge", **capm_regression(net_ridge - rf, exs)},
        {"model": "ar1_baseline", **capm_regression(net_ar1 - rf, exs)},
        {"model": "mean_forecast", **capm_regression(net_mean - rf, exs)},
    ])
    capm.to_csv(os.path.join(RESULTS, "capm.csv"), index=False)

    pd.DataFrame({
        "date": test["date"], "y_true": exs,
        "yhat_ridge": yhat_ridge, "yhat_ar1": yhat_ar1,
        "naive_forecast": naive_te,
    }).to_csv(os.path.join(RESULTS, "oos_predictions.csv"), index=False)

    pd.DataFrame({
        "date": test["date"], "ret_spy": fwd, "excess_spy": exs, "rf": rf,
        "ret_ridge": net_ridge, "ret_ar1": net_ar1,
        "ret_mean_forecast": net_mean, "ret_fixed_exposure": net_fixed,
        "pi_ridge": pi_ridge, "pi_ar1": pi_ar1,
    }).to_csv(os.path.join(RESULTS, "oos_strategy_returns.csv"), index=False)

    config = {
        "seed": SEED, "lambda_grid": LAMBDA_GRID, "lambda_selected": best_lam,
        "pi_max": PI_MAX, "cost_per_turnover": COST,
        "gamma_ridge": gamma_ridge, "gamma_ar1": gamma_ar1, "gamma_mean": gamma_mean,
        "mean_forecast_excess": yhat_mean,
        "train_end": TRAIN_END, "val_end": VAL_END,
        "features": FEATURES,
        "coefficients": dict(zip(FEATURES, ridge.coef_.round(6).tolist())),
        "intercept": float(ridge.intercept_),
        "test_diagnostics": {
            "avg_position_ridge": float(np.mean(pi_ridge)),
            "pct_days_long_ridge": float(np.mean(pi_ridge > 0)),
            "fixed_exposure_weight": float(pi_ridge.mean()),
        },
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
        },
    }
    with open(os.path.join(RESULTS, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("\n===== out-of-sample metrics (net of costs, excess-return accounting) =====")
    print(metrics.round(4).to_string(index=False))
    print("\n===== CAPM attribution (net excess on market excess, Newey-West) =====")
    print(capm.round(4).to_string(index=False))
    print(f"\navg position (ridge) = {pi_ridge.mean():.3f}, "
          f"long days = {100 * np.mean(pi_ridge > 0):.1f}%")


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
