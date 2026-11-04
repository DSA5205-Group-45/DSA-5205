"""Produce the figures required by the project brief (section 4.9):

  fig1_oos_r2.png             out-of-sample prediction quality (scatter + OOS R^2)
  fig2_cumulative_returns.png cumulative net-of-cost returns vs benchmarks (test)
  fig3_yearly_stats.png       annual return and Sharpe by calendar year (test)
  fig4_lambda_selection.png   validation Sharpe / OOS R^2 vs ridge lambda

All Sharpe ratios (including the yearly ones) use excess returns over the
3-month T-bill rate, consistent with the headline metrics table.
Inputs: output/results/*.csv produced by run_backtest.py.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "output", "results")
FIGS = os.path.join(ROOT, "output", "figures")
TRADING_DAYS = 252

plt.rcParams.update({"figure.dpi": 120, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})

with open(os.path.join(RESULTS, "run_config.json")) as _f:
    _FIXED_W = json.load(_f)["test_diagnostics"]["fixed_exposure_weight"]

MODELS = [  # (column prefix in oos_strategy_returns.csv, label, color, linestyle)
    ("ret_ridge", "ridge strategy (net)", "tab:blue", "-"),
    ("ret_mean_forecast", "mean forecast + vol mgmt (net)", "tab:green", "-"),
    ("ret_ar1", "AR(1) strategy (net)", "tab:orange", "-"),
    ("ret_fixed_exposure", f"fixed exposure {_FIXED_W:.2f} (exploratory)", "tab:red", "--"),
    ("ret_spy", "SPY buy-and-hold", "tab:gray", "-"),
]


def fig1(pred: pd.DataFrame, metrics: pd.DataFrame) -> None:
    r2_ridge = metrics.query("model=='ridge' and split=='test'")["oos_r2"].iloc[0]
    r2_ar1 = metrics.query("model=='ar1_baseline'")["oos_r2"].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))

    ax = axes[0]
    ax.scatter(pred["yhat_ridge"] * 100, pred["y_true"] * 100, s=4, alpha=0.25)
    b, a = np.polyfit(pred["yhat_ridge"], pred["y_true"], 1)
    xs = np.linspace(pred["yhat_ridge"].min(), pred["yhat_ridge"].max(), 50)
    ax.plot(xs * 100, (a + b * xs) * 100, color="red", lw=1.5, label=f"fit slope={b:.2f}")
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("ridge excess-return forecast (%, next holding day)")
    ax.set_ylabel("realized excess return (%)")
    ax.set_title(f"Test: forecast vs realized (OOS $R^2$={r2_ridge:.4f})")
    ax.legend()

    ax = axes[1]
    ax.bar(["ridge", "AR(1)"], [r2_ridge, r2_ar1], color=["tab:blue", "tab:orange"], width=0.5)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("OOS $R^2$ vs recursive historical mean (test)")
    ax.set_ylabel("OOS $R^2$")
    ax.margins(y=0.35)
    for i, v in enumerate([r2_ridge, r2_ar1]):
        ax.text(i, v + (0.00015 if v >= 0 else -0.00035), f"{v:.4f}", ha="center")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig1_oos_r2.png"))
    plt.close(fig)


def fig2(ret: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    for col, lab, c, ls in MODELS:
        ax.plot(ret["date"], np.cumprod(1 + ret[col]), label=lab, lw=1.2, color=c, ls=ls)
    ax.set_yscale("log")
    ax.set_ylabel("growth of $1 (log scale)")
    ax.set_title("Out-of-sample cumulative returns, net of costs (test period)")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    wealth = np.cumprod(1 + ret["ret_ridge"])
    dd = wealth / np.maximum.accumulate(wealth) - 1
    ax.fill_between(ret["date"], dd * 100, 0, color="tab:blue", alpha=0.4, label="ridge")
    wealth_bh = np.cumprod(1 + ret["ret_spy"])
    dd_bh = wealth_bh / np.maximum.accumulate(wealth_bh) - 1
    ax.plot(ret["date"], dd_bh * 100, color="tab:gray", lw=0.9, label="buy-and-hold")
    ax.set_ylabel("drawdown (%)"); ax.legend(loc="lower left")
    ax.set_xlabel("date")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig2_cumulative_returns.png"))
    plt.close(fig)


def fig3(ret: pd.DataFrame) -> None:
    df = ret.copy()
    df["year"] = df["date"].dt.year
    models = [("ret_ridge", "ridge", "tab:blue"),
              ("ret_mean_forecast", "mean fcst + vol mgmt", "tab:green"),
              ("ret_ar1", "AR(1)", "tab:orange"),
              ("ret_spy", "buy-and-hold", "tab:gray")]
    years, rows = sorted(df["year"].unique()), []
    for y in years:
        g = df[df["year"] == y]
        row = {"year": y}
        for col, key, _c in models:
            row[f"{key}_ret"] = np.prod(1 + g[col]) - 1
            exs = g[col] - g["rf"]  # excess-return Sharpe, same definition as the headline table
            row[f"{key}_sharpe"] = (exs.mean() / g[col].std(ddof=1) * np.sqrt(TRADING_DAYS)
                                    if g[col].std(ddof=1) > 0 else np.nan)
        rows.append(row)
    yr = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    n, width, x = len(models), 0.2, np.arange(len(yr))
    for k, (col, key, c) in enumerate(models):
        axes[0].bar(x + (k - (n - 1) / 2) * width, yr[f"{key}_ret"] * 100, width, label=key, color=c)
        axes[1].bar(x + (k - (n - 1) / 2) * width, yr[f"{key}_sharpe"], width, label=key, color=c)
    for ax, ttl in zip(axes, ["annual return (%)", "annualized Sharpe (excess over T-bill)"]):
        ax.set_xticks(x); ax.set_xticklabels(yr["year"], rotation=45)
        ax.axhline(0, color="k", lw=0.8); ax.set_title(ttl); ax.legend(fontsize=7.5)
    axes[0].set_ylabel("return (%)")
    fig.suptitle("Out-of-sample performance by calendar year (net of costs)", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig3_yearly_stats.png"), bbox_inches="tight")
    plt.close(fig)
    yr.to_csv(os.path.join(RESULTS, "yearly_stats.csv"), index=False)


def fig4(lam: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(6.5, 3.6))
    ax1.semilogx(lam["lambda"], lam["val_sharpe"], "o-", color="tab:blue", label="validation Sharpe")
    ax1.set_xlabel(r"ridge penalty $\lambda$ (log scale)")
    ax1.set_ylabel("validation Sharpe", color="tab:blue")
    best = lam.loc[lam["val_sharpe"].idxmax(), "lambda"]
    ax1.axvline(best, color="k", ls="--", lw=0.9)
    ax1.annotate(rf"selected $\lambda$={best:g}", xy=(best, lam["val_sharpe"].max()),
                 xytext=(0.45, 0.15), textcoords="axes fraction",
                 arrowprops={"arrowstyle": "->", "lw": 0.8})
    ax2 = ax1.twinx()
    ax2.semilogx(lam["lambda"], lam["val_oos_r2"], "s--", color="tab:red", label="validation OOS $R^2$")
    ax2.set_ylabel("validation OOS $R^2$", color="tab:red")
    ax2.grid(False)
    ax1.set_title("Validation-based model selection (2015-2019, single forward split)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig4_lambda_selection.png"))
    plt.close(fig)


def main() -> None:
    os.makedirs(FIGS, exist_ok=True)
    pred = pd.read_csv(os.path.join(RESULTS, "oos_predictions.csv"), parse_dates=["date"])
    ret = pd.read_csv(os.path.join(RESULTS, "oos_strategy_returns.csv"), parse_dates=["date"])
    metrics = pd.read_csv(os.path.join(RESULTS, "metrics.csv"))
    lam = pd.read_csv(os.path.join(RESULTS, "lambda_selection.csv"))

    fig1(pred, metrics)
    fig2(ret)
    fig3(ret)
    fig4(lam)
    print("[ok] figures written to", os.path.relpath(FIGS, ROOT))
    for f in sorted(os.listdir(FIGS)):
        print("   ", f)


if __name__ == "__main__":
    main()
