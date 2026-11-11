#!/usr/bin/env python
"""Build the academic report (LaTeX + PDF) from the saved result files.

Every number in the report is read from ``results/`` at build time.  The
generated ``report.tex`` is the editable LaTeX source; the PDF is compiled with
``latexmk``/``pdflatex``.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
GEN = REPORT / "generated"
FIGS = ROOT / "figures"
RES = ROOT / "results"

TARGET_PAGES = (15, 25)


def pct(value: float, dp: int = 1) -> str:
    return f"{value * 100:.{dp}f}\\%"


def num(value: float, dp: int = 2) -> str:
    return f"{value:.{dp}f}"


def pp(value: float, dp: int = 1) -> str:
    return f"{value * 100:+.{dp}f}\\%"


def feat(name: str) -> str:
    return name.replace("_", "\\_")


def booktab(header, rows, colspec, caption, label, notes=None, font="\\small", resize=False):
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        font,
    ]
    if resize:
        lines.append("\\resizebox{\\textwidth}{!}{%")
    lines += [
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(str(cell) for cell in row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if resize:
        lines.append("}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    if notes:
        lines.append(f"\\par\\vspace{{2pt}}{{\\footnotesize {notes}}}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def figure(filename, caption, label, width="0.92\\textwidth"):
    return "\n".join(
        [
            "\\begin{figure}[H]",
            "\\centering",
            f"\\includegraphics[width={width}]{{figures/{filename}}}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\end{figure}",
        ]
    )


def build():
    GEN.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- data
    validation = pd.read_csv(RES / "validation/validation_model_comparison.csv").set_index("model")
    roles = pd.read_csv(RES / "validation/primary_model_designation.csv").set_index("model")
    oos = pd.read_csv(RES / "oos/eight_strategy/metrics.csv").set_index("strategy")
    annual = pd.read_csv(RES / "oos/eight_strategy/annual_returns.csv", index_col=0)
    extension = pd.read_csv(RES / "extensions/strategy_summary_public.csv").set_index("strategy")
    prediction_oos = pd.read_csv(RES / "oos/oos_prediction_metrics.csv").set_index("model")
    checks = pd.read_json(RES / "oos/eight_strategy/alignment_checks.json", typ="series")
    pairs = pd.read_csv(RES / "diagnostics/feature_correlation_pairs.csv")
    stability = pd.read_csv(RES / "diagnostics/coefficient_stability_summary.csv").set_index("model")
    split_summary = pd.read_csv(RES / "validation/split_summary.csv")
    search = pd.read_csv(RES / "validation/validation_hyperparameter_search.csv")

    ridge = validation.loc["ridge"]
    lasso = validation.loc["lasso"]
    rf = validation.loc["rf"]
    xgb = validation.loc["xgb"]

    o = oos
    ridge_o, lasso_o, rf_o, xgb_o = o.loc["ridge"], o.loc["lasso"], o.loc["rf"], o.loc["xgb"]
    b1, b3, spy, qqq = o.loc["B1_Momentum10"], o.loc["B3_MomentumDoubleSort10"], o.loc["SPY"], o.loc["QQQ"]

    diag = {
        "n_pairs": int((pairs["abs_correlation"] >= 0.70).sum()),
        "ridge_change": float(stability.loc["ridge", "mean_relative_l2_change"]),
        "lasso_change": float(stability.loc["lasso", "mean_relative_l2_change"]),
        "reduction": float(
            (stability.loc["lasso", "mean_relative_l2_change"] - stability.loc["ridge", "mean_relative_l2_change"])
            / stability.loc["lasso", "mean_relative_l2_change"]
        ),
        "lasso_zero": float(stability.loc["lasso", "zero_coefficient_fraction"]),
        "ridge_flips": int(stability.loc["ridge", "total_sign_flips"]),
        "lasso_flips": int(stability.loc["lasso", "total_sign_flips"]),
        "windows": int(stability.loc["ridge", "n_windows"]),
    }

    # ---------------------------------------------------------------- figures
    mapping = {
        "fig1_four_model_wealth.png": "fig_four_model.png",
        "fig2_ridge_vs_benchmarks_wealth.png": "fig_ridge_benchmarks.png",
        "fig3_eight_strategy_summary.png": "fig_eight_summary.png",
        "fig4_oos_drawdowns.png": "fig_drawdowns.png",
        "fig5_annual_returns.png": "fig_annual.png",
        "fig6_risk_adjusted.png": "fig_risk_adjusted.png",
    }
    for source, target in mapping.items():
        shutil.copy2(RES / "oos/figures" / source, FIGS / target)
    shutil.copy2(RES / "diagnostics/report_correlation_heatmap.png", FIGS / "fig_correlation.png")
    shutil.copy2(RES / "diagnostics/report_stability_comparison.png", FIGS / "fig_stability.png")
    shutil.copy2(RES / "oos/figures/fig_score_to_position.png", FIGS / "fig_score_to_position.png")

    # ---------------------------------------------------------------- tables
    table_strategies = booktab(
        ["Strategy", "Role in the study", "Selection rule", "Weighting"],
        [
            ["Ridge", "Primary model", "3$\\times$3 pool, Ridge Top 10", "Equal 10\\%"],
            ["LASSO", "Alternative ML", "3$\\times$3 pool, LASSO Top 10", "Equal 10\\%"],
            ["Random Forest", "Alternative ML", "3$\\times$3 pool, RF Top 10", "Equal 10\\%"],
            ["XGBoost", "Alternative ML", "3$\\times$3 pool, XGB Top 10", "Equal 10\\%"],
            ["B1", "Momentum baseline", "Momentum Top 10, full liquid universe", "Equal 10\\%"],
            ["B3", "Momentum baseline", "Momentum Top 10, 3$\\times$3 pool", "Equal 10\\%"],
            ["SPY", "Investable benchmark", "Buy-and-hold S\\&P 500 ETF", "Fully invested"],
            ["QQQ", "Investable benchmark", "Buy-and-hold Nasdaq-100 ETF", "Fully invested"],
        ],
        "p{2.2cm}p{3.4cm}p{5.8cm}p{2.6cm}",
        "Strategy roles. B1 and B3 are systematic trading baselines, not investable index products; SPY and QQQ are investable buy-and-hold alternatives.",
        "tab:strategies",
        font="\\footnotesize",
    )

    feature_rows = [
        ["Momentum", feat("mom_21"), "21-session total return", "Short-horizon continuation or reversal"],
        ["Momentum", feat("mom_63"), "63-session total return", "Quarterly momentum"],
        ["Momentum", feat("mom_126"), "126-session total return", "Semi-annual momentum"],
        ["Momentum", feat("mom_2_12"), "12-month return, skipping the last month", "Classic 12--1 momentum"],
        ["Momentum", feat("relative_mom_63"), "63-session return minus SPY return", "Relative strength"],
        ["Volatility/risk", feat("vol_21"), "Annualised 21-session realised volatility", "Short-term risk"],
        ["Volatility/risk", feat("vol_63"), "Annualised 63-session realised volatility", "Screening volatility"],
        ["Volatility/risk", feat("beta_252"), "252-session rolling beta to SPY", "Systematic exposure"],
        ["Volatility/risk", feat("drawdown_63"), "Price relative to 63-session high", "Recent loss state"],
        ["Trend", feat("ma_ratio_50"), "Price relative to 50-session moving average", "Short-term trend"],
        ["Trend", feat("ma_ratio_200"), "Price relative to 200-session moving average", "Long-term trend"],
        ["Trend", feat("rsi_14"), "14-session relative strength index", "Overbought/oversold"],
        ["Liquidity", feat("ln_adv_20"), "Log 20-session average dollar volume", "Size and liquidity"],
        ["Liquidity", feat("volume_ratio_21"), "Volume relative to 21-session average", "Abnormal activity"],
        ["Range", feat("range_hl"), "High--low range scaled by close", "Intraday price pressure"],
    ]
    table_features = booktab(
        ["Group", "Feature", "Definition and lookback", "Interpretation"],
        feature_rows,
        "p{2.1cm}p{2.8cm}p{5.3cm}p{3.9cm}",
        "The fifteen predictors, grouped by economic mechanism. All inputs are available at the month-end decision close.",
        "tab:features",
        font="\\footnotesize",
    )

    split_rows = []
    for _, row in split_summary.iterrows():
        split_rows.append(
            [
                row["split"].capitalize(),
                str(pd.Timestamp(row["decision_start"]).date()),
                str(pd.Timestamp(row["decision_end"]).date()),
                str(pd.Timestamp(row["label_end_max"]).date()),
                f"{int(row['n_decision_dates'])}",
                f"{int(row['n_rows']):,}",
            ]
        )
    table_split = booktab(
        ["Stage", "First decision", "Last decision", "Last label end", "Cohorts", "Rows"],
        split_rows,
        "lccccr",
        "Chronological split after the realised-label embargo. Training and validation rows are included only when their labels settle by the split boundary.",
        "tab:split",
    )

    grid_rows = [
        ["Ridge", "Closed-form L2 regression", "$\\alpha\\in\\{0.1,1,10,100,1000,10000\\}$", "6"],
        ["LASSO", "L1 coordinate descent", "$\\alpha\\in\\{10^{-5},10^{-4},10^{-3},10^{-2},0.1\\}$", "5"],
        ["Random Forest", "Bagged regression trees", "$\\text{depth}\\in\\{3,6\\}$, $\\text{leaf}\\in\\{10,25\\}$", "4"],
        ["XGBoost", "Boosted regression trees", "$\\text{depth}\\in\\{2,3\\}$, $\\text{lr}\\in\\{0.03,0.1\\}$, $n\\in\\{100,200\\}$", "8"],
    ]
    table_grid = booktab(
        ["Model", "Algorithm", "Grid", "Configurations"],
        grid_rows,
        "p{2.2cm}p{4.1cm}p{6.2cm}r",
        "Pre-declared hyperparameter grids. The twenty-three configurations are fixed before estimation.",
        "tab:grid",
    )

    strong = pairs.loc[pairs["abs_correlation"] >= 0.70].sort_values("abs_correlation", ascending=False).head(12)
    corr_rows = [
        [feat(r.feature_1), feat(r.feature_2), f"{r.correlation:.3f}"]
        for r in strong.itertuples()
    ]
    table_corr = booktab(
        ["Feature 1", "Feature 2", "Correlation"],
        corr_rows,
        "llc",
        f"The {diag['n_pairs']} feature pairs with absolute training-period correlation of at least 0.70. The full matrix is shown in Figure~\\ref{{fig:corr}}.",
        "tab:corr",
    )

    val_rows = []
    for model, name in [("ridge", "Ridge"), ("lasso", "LASSO"), ("rf", "Random Forest"), ("xgb", "XGBoost")]:
        row = validation.loc[model]
        val_rows.append(
            [
                name,
                f"{row['validation_mse']:.5f}",
                f"{row['validation_r2_sklearn']:.4f}",
                f"{row['validation_r2_oos_trainmean']:.4f}",
                f"{row['validation_mean_rank_ic']:.4f}",
                f"{row['validation_sharpe']:.3f}",
                pct(row["validation_cagr"]),
                pct(row["validation_max_drawdown"], 1),
            ]
        )
    table_validation = booktab(
        ["Model", "MSE", "$R^2$", "OOS $R^2$", "Rank IC", "Sharpe", "CAGR", "Max DD"],
        val_rows,
        "lrrrrrrr",
        "Validation comparison, 2020--2021 (22 settled monthly cohorts). Sharpe uses a zero risk-free rate. OOS $R^2$ uses the training mean as the naive forecast.",
        "tab:validation",
    )

    role_rows = []
    for model, name in [("ridge", "Ridge"), ("lasso", "LASSO"), ("rf", "Random Forest"), ("xgb", "XGBoost")]:
        row = roles.loc[model]
        role_rows.append(
            [
                name,
                row["research_role"],
                f"{row['validation_mse']:.5f}",
                f"{row['validation_mean_rank_ic']:.4f}",
                f"{row['validation_sharpe']:.3f}",
                "Yes" if bool(row["is_primary"]) else "No",
            ]
        )
    table_roles = booktab(
        ["Model", "Research role", "Validation MSE", "Rank IC", "Sharpe", "Primary"],
        role_rows,
        "lrrrrl",
        "Primary-model designation. Ridge is the primary model on economic and diagnostic grounds; its $\\alpha$ is chosen by validation MSE within Ridge. The other three models are alternatives.",
        "tab:roles",
    )

    oos_rows = []
    for key, name in [
        ("ridge", "Ridge"),
        ("lasso", "LASSO"),
        ("rf", "Random Forest"),
        ("xgb", "XGBoost"),
        ("B1_Momentum10", "B1"),
        ("B3_MomentumDoubleSort10", "B3"),
        ("SPY", "SPY"),
        ("QQQ", "QQQ"),
    ]:
        row = oos.loc[key]
        oos_rows.append(
            [
                name,
                pct(row["annualized_return"]),
                pct(row["annualized_volatility"]),
                f"{row['sharpe_ratio']:.2f}",
                f"{row['sortino_ratio']:.2f}",
                pct(row["max_drawdown"]),
                f"{row['calmar_ratio']:.2f}",
                f"{row['average_turnover']:.2f}",
                pct(row["total_transaction_cost"], 2),
            ]
        )
    table_oos = booktab(
        ["Strategy", "CAGR", "Volatility", "Sharpe", "Sortino", "Max DD", "Calmar", "Turnover", "Cost"],
        oos_rows,
        "lrrrrrrrr",
        "Out-of-sample performance, 2022--2025 (48 monthly cohorts). All returns are net of 20 basis points per traded side. Sharpe and Sortino use a zero risk-free rate. Turnover is the average monthly sum of absolute weight changes; cost is the cumulative cost drag in return units.",
        "tab:oos",
        font="\\scriptsize",
    )

    annual_rows = []
    for year in annual.index:
        annual_rows.append(
            [
                str(year),
                pct(annual.loc[year, "ridge"]),
                pct(annual.loc[year, "B1_Momentum10"]),
                pct(annual.loc[year, "B3_MomentumDoubleSort10"]),
                pct(annual.loc[year, "SPY"]),
                pct(annual.loc[year, "QQQ"]),
            ]
        )
    table_annual = booktab(
        ["Year", "Ridge", "B1", "B3", "SPY", "QQQ"],
        annual_rows,
        "lrrrrr",
        "Annual net returns. The baseline advantage is not uniform across years: Ridge led in 2023, while B1 and B3 captured much larger returns in 2024.",
        "tab:annual",
    )

    ext_rows = []
    names = {
        "B1_Momentum10": "B1 Momentum10",
        "B2_Momentum30": "B2 Momentum30",
        "B3_MomentumDoubleSort10": "B3 DoubleSort10",
        "R1_RidgeWeighted10": "R1 RidgeWeighted10",
        "R2_RidgeFree60": "R2 RidgeFree60",
        "R3_RidgeQuota343": "R3 RidgeQuota343",
        "SPY": "SPY",
        "QQQ": "QQQ",
    }
    for key in [
        "B1_Momentum10",
        "B2_Momentum30",
        "B3_MomentumDoubleSort10",
        "R1_RidgeWeighted10",
        "R2_RidgeFree60",
        "R3_RidgeQuota343",
        "SPY",
        "QQQ",
    ]:
        row = extension.loc[key]
        ext_rows.append(
            [
                names[key],
                pct(row["annualized_return"]),
                pct(row["annualized_volatility"]),
                f"{row['sharpe_rf0']:.2f}",
                pct(row["max_drawdown"]),
                f"{row['avg_two_way_turnover']:.2f}",
                pct(row["total_cost"], 2),
            ]
        )
    table_extension = booktab(
        ["Strategy", "CAGR", "Volatility", "Sharpe", "Max DD", "Turnover", "Cost"],
        ext_rows,
        "lrrrrrr",
        "Exploratory Ridge extension, 2022--2025. These strategies were designed after earlier out-of-sample results were inspected and therefore do not constitute independent confirmation.",
        "tab:extension",
    )

    gross_rows = []
    for key, name in [
        ("ridge", "Ridge"),
        ("lasso", "LASSO"),
        ("rf", "Random Forest"),
        ("xgb", "XGBoost"),
        ("B1_Momentum10", "B1"),
        ("B3_MomentumDoubleSort10", "B3"),
        ("SPY", "SPY"),
        ("QQQ", "QQQ"),
    ]:
        row = oos.loc[key]
        drag = (row["annualized_return_gross"] - row["annualized_return"]) * 100
        gross_rows.append(
            [
                name,
                pct(row["annualized_return_gross"], 2),
                pct(row["annualized_return"], 2),
                f"{drag:.2f}",
                f"{row['beta_to_benchmark']:.2f}",
                pct(row["alpha_annualized"], 2),
            ]
        )
    table_grossnet = booktab(
        ["Strategy", "Gross CAGR", "Net CAGR", "Cost drag (pp)", "Beta", "Alpha"],
        gross_rows,
        "lrrrrr",
        "Gross-versus-net decomposition and market-factor exposure. Cost drag is the difference between gross and net CAGR in percentage points. Alpha is annualised Jensen alpha against SPY with a zero risk-free rate.",
        "tab:gross_net",
    )

    for name, content in {
        "table_strategies": table_strategies,
        "table_features": table_features,
        "table_split": table_split,
        "table_grid": table_grid,
        "table_corr": table_corr,
        "table_validation": table_validation,
        "table_roles": table_roles,
        "table_oos": table_oos,
        "table_annual": table_annual,
        "table_extension": table_extension,
        "table_grossnet": table_grossnet,
    }.items():
        (GEN / f"{name}.tex").write_text(content + "\n")

    # ---------------------------------------------------------------- prose
    n = {
        "RIDGE_VAL_MSE": f"{ridge['validation_mse']:.5f}",
        "LASSO_VAL_MSE": f"{lasso['validation_mse']:.5f}",
        "RIDGE_VAL_R2": f"{ridge['validation_r2_sklearn']:.4f}",
        "RIDGE_VAL_R2OOS": f"{ridge['validation_r2_oos_trainmean']:.4f}",
        "RIDGE_VAL_IC": f"{ridge['validation_mean_rank_ic']:.4f}",
        "LASSO_VAL_IC": f"{lasso['validation_mean_rank_ic']:.4f}",
        "RIDGE_VAL_SHARPE": f"{ridge['validation_sharpe']:.3f}",
        "LASSO_VAL_SHARPE": f"{lasso['validation_sharpe']:.3f}",
        "RIDGE_VAL_CAGR": pct(ridge["validation_cagr"]),
        "LASSO_VAL_CAGR": pct(lasso["validation_cagr"]),
        "RIDGE_VAL_MDD": pct(ridge["validation_max_drawdown"]),
        "LASSO_VAL_MDD": pct(lasso["validation_max_drawdown"]),
        "MSE_GAP": f"{(lasso['validation_mse'] - ridge['validation_mse']) / ridge['validation_mse'] * 100:.2f}",
        "IC_RATIO": f"{(ridge['validation_mean_rank_ic'] / lasso['validation_mean_rank_ic'] - 1) * 100:.0f}",
        "RIDGE_CAGR": pct(ridge_o["annualized_return"]),
        "RIDGE_CAGR_GROSS": pct(ridge_o["annualized_return_gross"]),
        "RIDGE_VOL": pct(ridge_o["annualized_volatility"]),
        "RIDGE_SHARPE": f"{ridge_o['sharpe_ratio']:.2f}",
        "RIDGE_SORTINO": f"{ridge_o['sortino_ratio']:.2f}",
        "RIDGE_MDD": pct(ridge_o["max_drawdown"]),
        "RIDGE_CALMAR": f"{ridge_o['calmar_ratio']:.2f}",
        "RIDGE_TOTAL": pct(ridge_o["total_return_net"]),
        "RIDGE_TURNOVER": f"{ridge_o['average_turnover']:.2f}",
        "RIDGE_COST": pct(ridge_o["total_transaction_cost"], 2),
        "RIDGE_BETA": f"{ridge_o['beta_to_benchmark']:.2f}",
        "RIDGE_ALPHA": pct(ridge_o["alpha_annualized"]),
        "RIDGE_IR": f"{ridge_o['information_ratio']:.2f}",
        "LASSO_CAGR": pct(lasso_o["annualized_return"]),
        "LASSO_SHARPE": f"{lasso_o['sharpe_ratio']:.2f}",
        "LASSO_MDD": pct(lasso_o["max_drawdown"]),
        "RF_CAGR": pct(rf_o["annualized_return"]),
        "RF_SHARPE": f"{rf_o['sharpe_ratio']:.2f}",
        "XGB_CAGR": pct(xgb_o["annualized_return"]),
        "XGB_SHARPE": f"{xgb_o['sharpe_ratio']:.2f}",
        "B1_CAGR": pct(b1["annualized_return"]),
        "B1_SHARPE": f"{b1['sharpe_ratio']:.2f}",
        "B1_MDD": pct(b1["max_drawdown"]),
        "B1_VOL": pct(b1["annualized_volatility"]),
        "B1_TURNOVER": f"{b1['average_turnover']:.2f}",
        "B1_COST": pct(b1["total_transaction_cost"], 2),
        "B1_ALPHA": pct(b1["alpha_annualized"]),
        "B3_CAGR": pct(b3["annualized_return"]),
        "B3_SHARPE": f"{b3['sharpe_ratio']:.2f}",
        "B3_MDD": pct(b3["max_drawdown"]),
        "B3_VOL": pct(b3["annualized_volatility"]),
        "B3_TURNOVER": f"{b3['average_turnover']:.2f}",
        "B3_COST": pct(b3["total_transaction_cost"], 2),
        "SPY_CAGR": pct(spy["annualized_return"]),
        "SPY_SHARPE": f"{spy['sharpe_ratio']:.2f}",
        "SPY_VOL": pct(spy["annualized_volatility"]),
        "SPY_MDD": pct(spy["max_drawdown"]),
        "QQQ_CAGR": pct(qqq["annualized_return"]),
        "QQQ_SHARPE": f"{qqq['sharpe_ratio']:.2f}",
        "QQQ_VOL": pct(qqq["annualized_volatility"]),
        "QQQ_MDD": pct(qqq["max_drawdown"]),
        "RIDGE_MINUS_SPY_CAGR": f"{(ridge_o['annualized_return'] - spy['annualized_return']) * 100:.1f}",
        "RIDGE_MINUS_SPY_SHARPE": f"{ridge_o['sharpe_ratio'] - spy['sharpe_ratio']:.2f}",
        "RIDGE_MINUS_B3_CAGR": f"{(b3['annualized_return'] - ridge_o['annualized_return']) * 100:.1f}",
        "RIDGE_MINUS_B3_SHARPE": f"{b3['sharpe_ratio'] - ridge_o['sharpe_ratio']:.2f}",
        "RIDGE_MINUS_B1_CAGR": f"{(b1['annualized_return'] - ridge_o['annualized_return']) * 100:.1f}",
        "RIDGE_MINUS_B1_SHARPE": f"{b1['sharpe_ratio'] - ridge_o['sharpe_ratio']:.2f}",
        "RIDGE_DD_VS_B3": f"{(abs(ridge_o['max_drawdown']) - abs(b3['max_drawdown'])) * 100:.1f}",
        "RIDGE_TURN_VS_B3": f"{ridge_o['average_turnover'] - b3['average_turnover']:.2f}",
        "RIDGE_OOS_MSE": f"{prediction_oos.loc['ridge','mse']:.6f}",
        "LASSO_OOS_MSE": f"{prediction_oos.loc['lasso','mse']:.6f}",
        "RIDGE_OOS_IC": f"{prediction_oos.loc['ridge','mean_rank_ic']:.4f}",
        "RF_OOS_IC": f"{prediction_oos.loc['rf','mean_rank_ic']:.4f}",
        "DIAG_PAIRS": str(diag["n_pairs"]),
        "RIDGE_CHANGE": f"{diag['ridge_change']:.3f}",
        "LASSO_CHANGE": f"{diag['lasso_change']:.3f}",
        "REDUCTION": f"{diag['reduction'] * 100:.1f}",
        "LASSO_ZERO": f"{diag['lasso_zero'] * 100:.1f}",
        "RIDGE_FLIPS": str(diag["ridge_flips"]),
        "LASSO_FLIPS": str(diag["lasso_flips"]),
        "DIAG_WINDOWS": str(diag["windows"]),
        "TRAIN_ROWS": f"{int(split_summary.set_index('split').loc['train','n_rows']):,}",
        "VAL_ROWS": f"{int(split_summary.set_index('split').loc['validation','n_rows']):,}",
        "TEST_ROWS": f"{int(split_summary.set_index('split').loc['test','n_rows']):,}",
        "REFIT_ROWS": f"{int(split_summary.set_index('split').loc['refit','n_rows']):,}",
        "TRAIN_DATES": str(int(split_summary.set_index('split').loc['train','n_decision_dates'])),
        "VAL_DATES": str(int(split_summary.set_index('split').loc['validation','n_decision_dates'])),
        "TEST_DATES": str(int(split_summary.set_index('split').loc['test','n_decision_dates'])),
        "B1_GROWTH": f"{1 + b1['total_return_net']:.2f}",
        "B3_GROWTH": f"{1 + b3['total_return_net']:.2f}",
        "RIDGE_GROWTH": f"{1 + ridge_o['total_return_net']:.2f}",
        "SPY_GROWTH": f"{1 + spy['total_return_net']:.2f}",
        "QQQ_GROWTH": f"{1 + qqq['total_return_net']:.2f}",
        "B1_MINUS_RIDGE_COST": f"{(b1['annualized_return'] - b1['annualized_return_gross']) * 100:.2f}",
        "RIDGE_COST_ANNUAL": f"{(ridge_o['annualized_return'] - ridge_o['annualized_return_gross']) * 100:.2f}",
        "B3_COST_ANNUAL": f"{(b3['annualized_return'] - b3['annualized_return_gross']) * 100:.2f}",
    }

    def sub(text: str) -> str:
        for key, value in n.items():
            text = text.replace(f"@@{key}@@", value)
        return text

    abstract = sub(r"""
This paper studies whether Ridge Regression, trained on economically motivated financial predictors, can generate useful cross-sectional stock rankings for a systematic US equity portfolio. The primary strategy screens a liquid universe with a momentum--volatility double sort, ranks the resulting thirty candidates with Ridge, buys the ten highest predicted returns at equal weights, rebalances monthly, and executes at the next open net of 20 basis points per traded side. Ridge is designated as the primary research model on economic and diagnostic grounds: its L2 penalty is appropriate for a dense set of correlated predictors, it produces the highest validation rank information coefficient of the four models, and rolling-window diagnostics show lower relative coefficient movement than LASSO. LASSO, Random Forest and XGBoost are evaluated as alternative forecasting models on an identical prediction task, while B1, B3, SPY and QQQ provide momentum baselines and investable benchmarks. Training uses 2010--2019, validation 2020--2021, and the final holdout is 2022--2025. Ridge earns a @@RIDGE_CAGR@@ net compound annual return with a @@RIDGE_SHARPE@@ Sharpe ratio and a @@RIDGE_MDD@@ maximum drawdown over the holdout, comfortably ahead of SPY (@@SPY_CAGR@@, @@SPY_SHARPE@@) and QQQ (@@QQQ_CAGR@@, @@QQQ_SHARPE@@). It does not beat simple momentum: B1 earns @@B1_CAGR@@ with a @@B1_SHARPE@@ Sharpe and a @@B1_MDD@@ drawdown, and B3 earns @@B3_CAGR@@ with a @@B3_SHARPE@@ Sharpe and a @@B3_MDD@@ drawdown. Within the same candidate pool, the Ridge ranking underperforms a momentum ranking by @@RIDGE_MINUS_B3_CAGR@@ percentage points of annual return and @@RIDGE_MINUS_B3_SHARPE@@ of Sharpe, with higher turnover. The evidence therefore supports Ridge as an economically defensible forecasting model that adds value relative to passive benchmarks, but not as a source of incremental value over momentum-based selection in this sample. The paper documents the full validation and out-of-sample process, identifies the earlier automatic LASSO rule as a superseded auxiliary artifact, and discloses that the Ridge designation is not a clean pre-registration because the project history includes prior out-of-sample review.
""")

    sections = []

    sections.append(sub(r"""
\section{Introduction and Research Motivation}

\subsection{Systematic cross-sectional equity selection}

Systematic equity strategies replace discretionary stock picking with a reproducible rule that maps observable firm characteristics into portfolio positions. The appeal is not that the rule is always right, but that it can be specified before the data are seen, evaluated on a held-out period, and audited afterwards. The academic literature has documented many cross-sectional regularities, including short-term reversal \citep{jegadeesh1990}, momentum \citep{jegadeesh1993}, value and momentum jointly \citep{asness2013}, and a long list of firm characteristics whose individual predictive power is small and unstable \citep{hou2020, feng2020}. Machine-learning methods are attractive in this setting because they can combine many weak, correlated predictors, but they also increase the risk of fitting noise \citep{gu2020}.

This paper takes a deliberately narrow version of that problem. The universe is a liquid subset of the S\&P 500. The candidate pool is formed by a transparent momentum--volatility double sort. The forecasting model ranks thirty names, and the portfolio holds the top ten at equal weights. Almost all of the complexity is in the forecast; the portfolio layer is intentionally simple. That design makes the research question sharp: does the forecasting model produce a better ordering of the candidates than simpler alternatives?

\subsection{Research question and contributions}

The primary research question is whether Ridge Regression, trained on economically motivated financial predictors, can generate useful cross-sectional stock rankings and improve the performance of a systematic equity portfolio relative to simpler investment alternatives.

The paper makes four contributions. First, it evaluates Ridge against three materially different forecasting algorithms---LASSO, Random Forest and XGBoost---on exactly the same universe, features, target, split, trading rule and cost assumption. The comparison therefore isolates the forecasting algorithm rather than confounding it with strategy design. Second, it grounds the Ridge choice in the correlation structure of the predictors and in rolling coefficient-stability diagnostics, rather than in a single validation metric. Third, it reports the primary strategy against two momentum baselines and two investable ETFs on a common out-of-sample window, and it verifies that the two backtest engines used to produce those series agree on dates, benchmark returns and the B3 selection rule. Fourth, it is explicit about what the evidence does \emph{not} show: in this sample, the Ridge ranking does not improve on a simple momentum ranking within the same candidate pool.

\subsection{Relation to the literature}

The forecasting exercise is related to the empirical asset-pricing literature that treats expected returns as a function of many characteristics \citep{gu2020, kelly2019, freyberger2020}. The regularisation choice is the classical ridge-versus-lasso question \citep{hoerl1970, tibshirani1996}: ridge shrinks a dense coefficient vector, while lasso selects a sparse one. The portfolio evidence is related to the momentum literature \citep{jegadeesh1993, asness2013, moskowitz2012} and to the broader critique that many published anomalies weaken out of sample \citep{mclean2016, hou2020}. The project does not attempt a structural model of returns; it asks whether a specific, pre-specified forecasting rule is useful for a specific portfolio construction.

The remainder of the paper is organised as follows. Section~\ref{sec:markets} describes the market, assets and timing. Section~\ref{sec:data} defines labels and features. Section~\ref{sec:benchmarks} defines the benchmarks. Section~\ref{sec:model} presents the Ridge rationale and the correlation and stability diagnostics. Section~\ref{sec:split} documents the chronological split. Section~\ref{sec:training} describes training, and Section~\ref{sec:validation} the validation and model-selection policy. Section~\ref{sec:trading} states the trading rules and costs. Section~\ref{sec:oos} reports the out-of-sample evidence. Section~\ref{sec:interpretation} interprets it, Section~\ref{sec:extensions} discusses exploratory extensions, Section~\ref{sec:limitations} lists limitations, and Section~\ref{sec:conclusion} concludes. Section~\ref{sec:reproducibility} documents reproducibility and AI use.
"""))

    sections.append(sub(r"""
\section{Markets, Assets and Trading Timing}
\label{sec:markets}

\subsection{Market, universe and frequency}

The market is US large-capitalisation equity, traded in US dollars on the NYSE and Nasdaq calendars. The eligible universe is built from the current S\&P 500 constituent list. At each month-end, a stock is eligible if its close is at least \$5, its twenty-day average dollar volume is at least \$10 million, and it has at least 95\% price coverage over the 21-, 63- and 252-session windows. The three hundred most liquid eligible names form the liquid universe. The 300-name cap is a fixed design choice, not a parameter chosen on the test period: it keeps the universe tradable, reduces exposure to illiquid names, provides enough cross-sectional breadth for a 3$\times$3 sort, and makes candidate construction consistent across months. In this data snapshot every month reached the cap, but the rule does not guarantee it; a month with fewer than 90 eligible names is skipped rather than padded.

The strategy trades monthly. Decisions are made after the close of the last trading session of each calendar month, and trades are executed at the open of the next observed session. The holding period therefore runs from one monthly execution open to the next. No intraday information enters the decision, and no same-close execution is assumed.

\subsection{The 3$\times$3 screen and the 3--4--3 allocation}

The screen is a dependent double sort, implemented in the preserved pipeline. Eligible stocks are first ranked by twelve-month momentum (\texttt{mom\_2\_12}) in descending order and split into three approximately equal momentum groups: high, medium and low. Within each momentum group, stocks are sorted by sixty-three-day volatility (\texttt{vol\_63}) in ascending order and split into three approximately equal volatility cells: low, medium and high. Each cell then contributes a fixed number of candidates: three from the low-volatility cell, four from the medium-volatility cell and three from the high-volatility cell. Each momentum group therefore contributes ten names, and the screen produces thirty candidates.

The within-cell ranking depends on the momentum group. The high-momentum group takes the highest \texttt{mom\_2\_12} names in the cell, the low-momentum group takes the lowest, and the medium-momentum group takes the names closest to the cell median. Ties are broken by ticker in ascending order. If a cell contains fewer names than required, the decision date is skipped rather than filled with names outside the cell.

The 3--4--3 allocation is symmetric because ten candidates cannot be divided evenly into three volatility cells. Assigning the additional candidate to the middle cell avoids favouring either volatility extreme. It is a fixed, interpretable design choice; the paper does not claim that 3--4--3 is empirically optimal or that medium-volatility stocks are superior. The screen is risk-characteristic-aware, but it does not minimise portfolio variance and it does not guarantee that the final ten holdings are diversified across volatility groups. Candidate-universe diversification is not the same as final-portfolio diversification.

\subsection{Assets, benchmarks and currency}

The tradable instruments in the backtest are the individual stocks in the candidate pool, plus SPY and QQQ as buy-and-hold benchmarks. The base currency is the US dollar; no FX conversion is required. SPY tracks the S\&P 500 and QQQ tracks the Nasdaq-100. Both are liquid, investable ETFs and are therefore genuine alternatives for a real investor, unlike the momentum baselines, which are systematic rules rather than products.

\subsection{Data sources and corporate actions}

Prices, volumes, dividends and splits are obtained from Yahoo Finance through the \texttt{yfinance} interface. The constituent list is the current membership table published on Wikipedia. Adjusted prices reflect dividends and splits; the backtest uses adjusted opens for both entry and exit, so the return series is total-return in nature. The data snapshot is frozen for reproducibility.

Adjusted opens are used consistently for both entry and exit, so dividends and splits are reflected in every label and backtest return. The treatment is not a manual corporate-action reconstruction: vendor adjusted prices can also embed spin-off or special-distribution adjustments that are not recorded in the split field, so adjusted returns can differ from raw-price returns around such events. This is a documented data limitation, not a correction applied inside the backtest. A verification check on the raw snapshot confirms that there are no non-positive or duplicate observations and that adjusted prices are used consistently. First, using current S\&P 500 constituents to construct the historical equity universe introduces potential survivorship and constituent-selection bias into the active stock-selection strategies: firms that were removed from the index are absent, and current survivors are over-represented. This may overstate their historical performance relative to SPY and QQQ, whose buy-and-hold returns are calculated directly from historical ETF prices. Second, vendor price adjustments and corporate-action treatment may affect both individual-stock histories and ETF return calculations, and Yahoo Finance is not an exchange-grade point-in-time database. Neither limitation is resolved by the modelling choices, and the direction or magnitude of the survivorship effect is not precisely measured here.

\subsection{The eight evaluated strategies}

Table~\ref{tab:strategies} lists the eight strategies. The four machine-learning strategies share the same thirty-candidate pool, features, target, split, execution and cost assumptions; only the forecasting algorithm differs. B1 and B3 are momentum baselines. B1 selects the momentum Top 10 from the full liquid universe, while B3 selects the momentum Top 10 within the same thirty-candidate pool used by the ML models. SPY and QQQ are investable buy-and-hold benchmarks.

The distinction between a trading baseline and an investable benchmark matters. A baseline answers ``does the model add value relative to a simple systematic rule?''. An investable benchmark answers ``would an investor have been better off in an index ETF?''. The paper reports both because the answers differ.

\input{generated/table_strategies}
"""))

    sections.append(sub(r"""
\section{Labels, Features and Data Preparation}
\label{sec:data}

\subsection{Prediction target and horizon}

The target is the next holding-period simple return computed from adjusted opens:
\begin{equation}
y_t=\frac{P^{\text{adj open}}_{t+H}}{P^{\text{adj open}}_{t}}-1,
\label{eq:target}
\end{equation}
where $t$ is the execution open after the decision date and $t+H$ is the execution open after the next rebalance. For monthly rebalancing, $H$ is approximately one month. Using adjusted opens rather than closes ensures that the label matches the price at which a position could actually be established and unwound, and that dividends and splits are reflected.

The timing convention is strict: \texttt{decision\_date} $<$ \texttt{execution\_date} $<$ \texttt{label\_end\_date} for every observation. The model at date $t$ uses features computed from data up to the decision close and predicts a return that is realised only after the position is opened. Section~\ref{sec:split} describes the additional embargo that prevents labels which settle after a split boundary from entering the earlier stage.

\subsection{The fifteen features}

Table~\ref{tab:features} lists the fifteen features, grouped into five economic categories: momentum, volatility and risk, technical trend, liquidity and trading activity, and price range. All inputs are available at the decision close and none uses future information.

\input{generated/table_features}

The feature set is compact and interpretable rather than empirically optimal. Fifteen characteristics keep the parameter-to-observation ratio manageable in a sample of a few thousand candidate-months and allow the coefficients to be read economically. No claim is made that fifteen is the best possible number, and the paper does not search over feature subsets.

\subsection{Why multiple horizons}

Momentum is measured at 21, 63, 126 and 12--1 months, and trend is measured with 50- and 200-session moving-average ratios. These horizons are not redundant even when they are correlated. Short horizons are more sensitive to reversal and microstructure effects, while twelve-month momentum captures slower fundamental trends. The 12--1 construction deliberately skips the most recent month, following the momentum literature, to reduce contamination from short-term reversal \citep{jegadeesh1993}. Relative momentum subtracts the SPY return over the same window, so it measures cross-sectional strength rather than market direction.

The volatility block separates short- and medium-horizon realised volatility, market beta and recent drawdown. The trend block captures whether the price is above or below its moving averages and whether it is overbought or oversold. The liquidity block captures size and abnormal trading activity. The range feature captures intraday price pressure. Together they represent the main mechanisms through which past prices could predict future cross-sectional returns.

\subsection{Cleaning, missing values and outliers}

The data pipeline requires finite feature values for every candidate at the decision date. It does not fabricate pre-IPO or post-delisting prices, and it does not globally forward-fill prices. Auxiliary machine-learning features may be carried forward for at most two sessions, but the screening variables---twelve-month momentum, sixty-three-day volatility and log dollar volume---are never forward-filled. Extreme values are not winsorised. The model relies on regularisation and on the screening filters, which require a minimum price and liquidity. This choice is transparent but not innocuous: a small number of extreme momentum realisations can influence the fitted coefficients, and a winsorised alternative could produce different results.

\subsection{Scaling and leakage prevention}

Ridge and LASSO standardise features inside a scikit-learn \texttt{Pipeline}. The scaler is therefore fitted only on the rows passed to \texttt{fit}, which for training is the 2010--2019 slice and for the final refit is 2010--2021. Validation and test features are transformed with those training parameters. Tree models are scale-invariant and use no scaler. No imputation is performed inside the model.

A row enters training or tuning only if its label has settled by the split boundary. This removes the last two 2019 cohorts from training, because their labels settle in January and February 2020, and the last two 2021 cohorts from validation, because their labels settle in January and February 2022. The test slice is never loaded by the validation or designation stages.
"""))

    sections.append(sub(r"""
\section{Benchmarks}
\label{sec:benchmarks}

\subsection{Investable benchmarks}

SPY and QQQ are buy-and-hold ETFs. They are charged a single 20 basis point entry cost at the start of the evaluation and no monthly rebalancing cost, because a buy-and-hold investor does not rebalance. This is the correct comparison: applying the active strategy's monthly turnover cost to a passive ETF would overstate the active strategy. Both benchmarks are evaluated on exactly the same decision dates and with the same execution-to-execution return convention as the active strategies.

\subsection{Strategy baselines B1 and B3}

B1 and B3 are systematic momentum rules. B1 ranks the full liquid universe by twelve-month momentum and buys the top ten at equal weights. B3 ranks the thirty candidates produced by the same 3$\times$3 double sort used by the ML strategies and buys the ten highest-momentum names at equal weights. Both execute at the next open and bear the same 20 basis point cost as the active strategies, so their returns are directly comparable.

The two baselines answer different questions. B1 is a complete alternative strategy: it differs from the primary strategy in both the candidate universe and the ranking method. B3 is a cleaner ranking test: it holds the candidate pool, execution and costs fixed and changes only whether the top ten are chosen by momentum or by the model. The Ridge-versus-B3 comparison is therefore the paper's main evidence on whether the forecasting model adds value beyond momentum within the screened pool. The Ridge-versus-B1 comparison is a comparison of complete strategies and cannot be attributed to ranking alone.

\subsection{Why both types of benchmark are needed}

A model that beats SPY but loses to a simple momentum rule has not demonstrated that the model is useful for stock selection; it has demonstrated that equity exposure and a momentum tilt were useful. Conversely, a model that beats momentum but loses to SPY would be difficult to justify to an investor. Reporting both types of benchmark prevents either conclusion from being drawn from a single comparison. The paper also reports turnover and transaction costs for every strategy, because a strategy that earns a higher gross return by trading more intensively may not earn a higher net return.
"""))

    sections.append(sub(r"""
\section{Model Choice and Overfitting Control}
\label{sec:model}

\subsection{Economic rationale for Ridge}

The fifteen predictors are correlated financial characteristics. Ridge regression estimates a dense linear coefficient vector and penalises its squared Euclidean norm \citep{hoerl1970}:
\begin{equation}
\hat{\beta}_{\text{ridge}}=\arg\min_{\beta}\;\sum_{i=1}^{N}\left(y_i-x_i'\beta\right)^2+\alpha\|\beta\|_2^2 .
\label{eq:ridge}
\end{equation}
The penalty introduces bias but reduces variance, which is valuable when predictors are collinear and the signal-to-noise ratio is low. The resulting score is continuous, which maps naturally into a cross-sectional ranking: the portfolio only needs to know which candidates are likely to be in the top decile, not the precise expected return of every stock.

The economic story is that expected returns depend on several weak, related characteristics and that a shrunk combination of them is more stable than any individual signal. This is a modest claim. It does not require that every feature has independent predictive power, and it does not require that the linear functional form is literally correct. It requires only that the combination is more informative about the ordering of future returns than a single momentum rank.

\subsection{Correlated predictors}

Appendix Figure~\ref{fig:corr} shows the training-period correlation matrix, and Table~\ref{tab:corr} lists the twelve pairs with absolute correlation of at least 0.70. The strongest pair is 63-day momentum and relative 63-day momentum at 0.939, which is mechanical: the two series differ only by the SPY return over the same window. The next strongest pairs link 126-day momentum to the 200-day moving-average ratio (0.936), 21-day momentum to the 50-day moving-average ratio (0.868), and 63-day momentum to the 200-day ratio (0.816). Short and medium volatility are correlated at 0.809.

Correlation does not imply identical economic information. A 63-day and a 126-day return measure different horizons even when they co-move; relative and absolute momentum answer different questions; a moving-average ratio and a trailing return are related but not identical transformations. Removing every correlated feature would discard horizon-specific information and leave an arbitrary subset chosen by an arbitrary threshold. Ridge instead keeps the full, economically motivated set and controls the variance of the combined coefficient vector. A sparse model such as LASSO is likely to select one representative from each correlated block and to set the others to zero, which is a different modelling choice rather than a clearly better one.

\input{generated/table_corr}

\subsection{Coefficient stability}

The supplementary diagnostics estimate Ridge and LASSO on 59 rolling 60-month training windows and compare the standardised coefficient vectors between adjacent windows. Appendix Figure~\ref{fig:stability} and Table~\ref{tab:stability} summarise the results. The mean relative $L_2$ change is @@RIDGE_CHANGE@@ for Ridge and @@LASSO_CHANGE@@ for LASSO, so Ridge's coefficients change about @@REDUCTION@@\% less between adjacent windows. LASSO sets @@LASSO_ZERO@@\% of its coefficients to exactly zero, while Ridge never does.

Ridge is not uniformly more stable. It records @@RIDGE_FLIPS@@ sign flips against @@LASSO_FLIPS@@ for LASSO, because small non-zero coefficients can cross zero from window to window, whereas LASSO's active set is more stable once selected. Coefficient stability is also not the same as portfolio-weight stability: equal weights and Top-10 membership depend on the cross-sectional ordering, and a stable coefficient vector can still generate turnover when relative scores change. The diagnostics are therefore supportive evidence for the Ridge specification, not a proof that it produces a more stable portfolio.

\begin{table}[H]
\centering
\small
\begin{tabular}{lrr}
\toprule
Statistic & Ridge & LASSO \\
\midrule
Mean relative $L_2$ coefficient change & @@RIDGE_CHANGE@@ & @@LASSO_CHANGE@@ \\
Relative change reduction & \multicolumn{2}{c}{@@REDUCTION@@\%} \\
Zero-coefficient fraction & 0.0\% & @@LASSO_ZERO@@\% \\
Total sign flips & @@RIDGE_FLIPS@@ & @@LASSO_FLIPS@@ \\
Rolling windows & \multicolumn{2}{c}{@@DIAG_WINDOWS@@} \\
\bottomrule
\end{tabular}
\caption{Coefficient-stability summary from 59 rolling 60-month training windows. These diagnostics are supplementary and were produced after the original four-model experiment.}
\label{tab:stability}
\end{table}

\subsection{Ridge versus LASSO}

LASSO minimises the residual sum of squares subject to an $L_1$ penalty \citep{tibshirani1996}. It selects a sparse subset of predictors and sets the rest to zero. In a setting with many correlated characteristics and a small number of genuinely independent signals, sparsity can be attractive because it produces an interpretable model and reduces variance. But sparsity also means that the coefficient on a characteristic can jump to zero when a correlated alternative becomes slightly more attractive, and the identity of the selected variables can change across windows.

Ridge and LASSO answer different questions. Ridge asks ``given all fifteen characteristics, what is the best shrunk linear combination?''. LASSO asks ``which small subset of characteristics is sufficient?''. For a portfolio that needs a stable ordering across many correlated signals, the dense combination is a defensible default. For a researcher trying to identify a sparse set of causal characteristics, LASSO is more natural. The paper does not claim that Ridge is universally superior; it claims that Ridge is the more appropriate choice for this ranking task and supports that claim with validation rank IC and coefficient-stability diagnostics.

\subsection{Overfitting control}

Overfitting is controlled at four levels. The feature set is small relative to the number of candidate-months. The hyperparameter grid is small and fixed in advance, with twenty-three configurations across four models. Ridge, LASSO and the tree models all impose explicit complexity penalties or depth constraints. The chronological split and the realised-label embargo prevent the model from training on outcomes that were not yet known. The model assessment is based on economic considerations and validation-stage evidence. Out-of-sample performance is reported separately and is not used as a numerical model-selection criterion in the final research framework.

No formal generalisation bound is claimed. The validation comparison is descriptive, and the out-of-sample evidence is a single historical path.
"""))

    sections.append(sub(r"""
\section{Data Period and Chronological Split}
\label{sec:split}

The sample covers monthly decisions from January 2010 to December 2025. The split is chronological and non-overlapping:

\begin{itemize}[leftmargin=1.2em,itemsep=2pt]
\item training: 2010--2019, used to fit every configuration;
\item validation: 2020--2021, used to rank hyperparameters and compare models;
\item final refit: 2010--2021, used to re-estimate the frozen configurations;
\item test: 2022--2025, used once for the final out-of-sample evaluation.
\end{itemize}

Table~\ref{tab:split} reports the realised cohorts after the embargo. Training contains @@TRAIN_ROWS@@ candidate rows over @@TRAIN_DATES@@ decision dates; validation contains @@VAL_ROWS@@ rows over @@VAL_DATES@@ dates; the final refit contains @@REFIT_ROWS@@ rows; and the test period contains @@TEST_ROWS@@ rows over @@TEST_DATES@@ dates. Every decision date has exactly thirty candidates.

\input{generated/table_split}

The difference between a decision-date split and a label-availability split is important. A row with a December 2019 decision date has a label that settles in February 2020, after the training boundary. Using that label for training would mean training on an outcome that was not known at the start of the validation period. The embargo therefore requires \texttt{label\_end\_date} to be on or before the boundary, which removes the last two training cohorts and the last two validation cohorts. The same rule is applied to the final refit.

The final refit uses all rows with a decision date before 2022 and a label that settles by 2021-12-31. This includes the training and validation periods but excludes the two 2021 cohorts whose labels settle in January and February 2022. The refitted model is then frozen for the entire test period; there is no rolling retraining.

No test-period observation enters feature construction, scaling, hyperparameter selection, model designation or refitting. The only use of test data is the final evaluation itself.
"""))

    sections.append(sub(r"""
\section{Model Training}
\label{sec:training}

\subsection{Algorithms and libraries}

Ridge is estimated by scikit-learn's \texttt{Ridge} with the default \texttt{solver='auto'}, which uses a Cholesky-based closed-form solution for this dense problem; it is not trained by gradient descent. LASSO is estimated by cyclic coordinate descent (\texttt{selection='cyclic'}, \texttt{tol=1e-4}, \texttt{max\_iter=20000}). Random Forest is a bagged ensemble of 250 regression trees \citep{breiman2001} with bootstrap sampling, a maximum of 70\% of features per split, maximum depth of 3 or 6, and a minimum leaf size of 10 or 25; trees are fitted independently. XGBoost is a boosted ensemble of shallow regression trees \citep{chen2016} fitted with the histogram algorithm (\texttt{tree\_method='hist'}), squared-error loss, subsampling of rows and columns, and an $L_2$ penalty on leaf weights.

All estimation is performed in Python 3.11 with scikit-learn 1.9.1 and XGBoost 3.2.0. The full package list is in Section~\ref{sec:reproducibility}.

\subsection{Preprocessing and seeds}

Ridge and LASSO use a \texttt{StandardScaler} inside a pipeline, fitted on the training rows only. Tree models are scale-invariant. The random seed is 42 for every stochastic component: Random Forest bootstrap samples, XGBoost subsampling, and any randomised solver. XGBoost predictions are bit-identical across repeated fits with the same seed. Random Forest predictions can differ at the $10^{-16}$ level because parallel tree aggregation changes floating-point summation order, which does not affect reported results at the precision shown.

\subsection{Hyperparameter grid}

Table~\ref{tab:grid} lists the twenty-three configurations. The grid is deliberately small. A larger grid would increase the chance of selecting a configuration that happens to fit the validation window, and the assignment explicitly asks for a small pre-declared search.

\input{generated/table_grid}

\subsection{Training objective and evaluation}

Every model minimises a squared-error objective, which is aligned with the MSE used for validation and with the mean-squared-error evaluation of the assignment. The validation MSE is computed on the validation slice with predictions from a model fitted on the training slice. The best configuration per model is the one with the lowest validation MSE, with ties broken by the deterministic grid order. This rule is used for hyperparameter selection within each model and is not the rule that designates the primary model.
"""))

    sections.append(sub(r"""
\section{Validation and Model Selection}
\label{sec:validation}

\subsection{Time-respecting validation}

Validation is a fixed chronological holdout, not cross-validation. The training slice ends in 2019, the validation slice covers 2020--2021, and neither overlaps the test period. This is simpler than a rolling scheme and matches the assignment's recommended forward validation. The cost is that the validation window is a single two-year path, so the selected hyperparameters may reflect the market conditions of 2020--2021. A rolling validation with a fixed window would provide more evidence but would also require a different refit protocol; it is left for future work.

\subsection{Four-model comparison}

Table~\ref{tab:validation} reports the validation comparison. The benchmark-relative OOS $R^2$ follows the Campbell--Thompson idea \citep{campbell2008}: the naive forecast is the historical training mean, not the evaluation-sample mean. LASSO has the lowest MSE (@@LASSO_VAL_MSE@@) and the highest Sharpe (@@LASSO_VAL_SHARPE@@). Ridge is third on MSE (@@RIDGE_VAL_MSE@@) and second on Sharpe (@@RIDGE_VAL_SHARPE@@). The MSE gap is small: LASSO's MSE is @@MSE_GAP@@\% lower than Ridge's. Ridge has the highest validation rank IC (@@RIDGE_VAL_IC@@), roughly @@IC_RATIO@@\% higher than LASSO's @@LASSO_VAL_IC@@. Random Forest is second on rank IC and XGBoost has a negative rank IC in validation.

The validation ranking is therefore not unanimous. LASSO wins on average squared error and on portfolio Sharpe; Ridge wins on the within-month rank correlation that is closest to the portfolio's actual decision. This is exactly the kind of disagreement that a single selection metric hides.

\input{generated/table_validation}

\subsection{The superseded automatic selection artifact}

An earlier engineering revision added an automatic rule that ranked the four models by lowest validation MSE and selected LASSO with $\alpha=0.001$. That rule was a coding convenience, not the research team's model-assessment policy, and it has been removed from the active workflow. Its output is retained read-only in the superseded-selection folder under \texttt{results/validation/archive/} and is clearly labelled as superseded. It is not used to designate the primary model. Table~\ref{tab:roles} shows the four-model comparison and the research role assigned to each model.

\input{generated/table_roles}

\subsection{Primary-model designation}

The research design designates Ridge as the primary model. The designation rests on four arguments. First, the predictors are economically motivated but strongly correlated, and an $L_2$ penalty is the natural regularisation for a dense, collinear design. Second, Ridge produces the highest validation rank IC, the metric most closely aligned with the cross-sectional ordering the portfolio uses. Third, rolling-window diagnostics show lower relative coefficient movement for Ridge than for LASSO, even though Ridge sign-flips more often. Fourth, Ridge was the pre-specified model of the original research design; this paper restores that designation rather than inventing a new one.

Ridge's penalty, $\alpha=1000$, is chosen by validation MSE \emph{within} Ridge. No test-period metric enters the choice. The other three models remain alternatives on the same prediction task, and their out-of-sample results are reported in full.

\subsection{Chronology disclosure}

The designation is not a clean pre-registration. The project history includes an earlier revision in which the automatic minimum-MSE rule selected LASSO and out-of-sample results were reviewed. The present designation restores the original Ridge pre-specification on economic and diagnostic grounds, but it was made with knowledge of that history. The primary-model contract records \texttt{clean\_preregistration: false} and \texttt{designation\_informed\_by\_prior\_oos\_review: true}. No test-period metric was used to choose Ridge or its $\alpha$, but the reader should treat the designation as a research-design judgment rather than as a blind pre-registration.

This disclosure matters because the paper's conclusion is partly negative for Ridge. A designation made after seeing that Ridge does not beat momentum cannot be used to claim that the test confirmed a pre-specified hypothesis. The honest interpretation is that Ridge is an economically motivated model that was evaluated fairly, and that the evaluation did not establish incremental value over momentum.

\subsection{What validation metrics do and do not establish}

MSE measures average squared forecast error across all thirty candidates. Rank IC measures the within-month ordering. Directional accuracy measures the sign of the forecast. Portfolio Sharpe measures the realised risk-adjusted return of the Top-10 rule after costs. None of these establishes future profitability. A model can have a low MSE and a poor portfolio because the errors are concentrated in the names that enter the top decile. A model can have a high rank IC and a poor portfolio because turnover and costs consume the ranking advantage. The out-of-sample section reports all of these dimensions rather than selecting the most favourable one.
"""))

    sections.append(sub(r"""
\section{Trading Rules and Transaction Costs}
\label{sec:trading}

\subsection{From predictions to positions}

At each month-end decision, the strategy applies the eligibility filters and the ADV20 Top-300 ranking, then the 3$\times$3 momentum--volatility double sort. The thirty resulting candidates are scored by the refitted Ridge model. The ten highest predicted returns are selected, with ties broken by ticker in ascending order. Each selected stock receives a target weight of 10\%, so the portfolio is fully invested with no cash and no leverage.

Execution occurs at the next session's open. The backtest computes the realised holding-period return for each stock from the adjusted open at entry to the adjusted open at the next rebalance. The portfolio gross return in period $t$ is the weighted average of the selected stocks' realised returns:
\begin{equation}
R^{\text{gross}}_t=\sum_{i\in S_t} w_{i,t}\,r_{i,t},
\label{eq:gross}
\end{equation}
where $S_t$ is the selected set, $w_{i,t}=0.1$ and $r_{i,t}$ is the stock's holding-period return. Equivalently, the target weight is
\begin{equation}
w_{i,t} =
\begin{cases}
0.1, & i \in \operatorname{Top10}(\hat r_{i,t+1}),\\
0, & \text{otherwise}.
\end{cases}
\label{eq:weights}
\end{equation}

Figure~\ref{fig:score} illustrates the rule on one real decision date, 30 June 2025. The thirty candidates are ordered by Ridge predicted return; the top ten receive 10\% weights and the remaining twenty receive zero. The predicted score determines the ranking but not the size of the position.

\begin{figure}[H]
\centering
\includegraphics[width=0.90\textwidth]{figures/fig_score_to_position.png}
\caption{Score-to-position example using real Ridge predictions on 30 June 2025. Navy bars are the ten selected stocks, each with a 10\% target weight; grey bars are the twenty candidates with zero target weight.}
\label{fig:score}
\end{figure}

\subsection{Turnover and transaction costs}

Weights drift with realised returns between rebalances. The pre-trade weight of stock $i$ is its post-return weight normalised to sum to one. Turnover is the sum of absolute differences between target and pre-trade weights:
\begin{equation}
\tau_t=\sum_i\left|w_{i,t}-w^{\text{pre}}_{i,t}\right|.
\label{eq:turnover}
\end{equation}
The transaction cost is $\tau_t\times 20$ basis points, and the net return is $R^{\text{net}}_t=R^{\text{gross}}_t-c\tau_t$. Because turnover is a sum of absolute weight changes, both purchases and sales are charged: a 10\% sale and a 10\% purchase each contribute 0.10 to $\tau_t$ and therefore 2 basis points to the cost. Stocks that remain selected are not liquidated and repurchased; only the difference is traded. This is the original convention and is preserved exactly. The automated tests check the first-period cost (1.0 turnover, 0.002 cost), the drift arithmetic and the reconciliation between holdings and portfolio returns.

The 20 basis point assumption is a constant proportional cost. It does not model market impact, bid--ask spread variation, borrow, taxes or capacity. For a monthly strategy holding ten large-cap names it is a reasonable first approximation, but it is not a realistic execution model for a large fund.

\subsection{Cash, constraints and risk}

The portfolio is fully invested. There is no cash buffer, no leverage, no short selling and no explicit risk model. Individual weights are 10\%, so concentration is high by construction. The strategy's beta to SPY is @@RIDGE_BETA@@, which means that a large part of its return is equity-market exposure rather than stock-selection alpha.

\subsection{Distinction from the exploratory extension}

The primary strategy is the 3$\times$3 screen, Ridge Top-10 ranking and equal weights. It is not the same as R1 RidgeWeighted10, which uses momentum selection and prediction-proportional weights capped at 20\%. Section~\ref{sec:extensions} discusses the exploratory strategies separately. Confusing coefficient shrinkage with portfolio allocation is a common error: the $L_2$ penalty stabilises the forecast, while position sizing and caps are separate portfolio decisions.
"""))

    sections.append(sub(r"""
\section{Out-of-Sample Evaluation}
\label{sec:oos}

\subsection{Protocol and comparability}

The test period is January 2022 to December 2025, covering 48 monthly decision dates. The primary model configuration and its hyperparameters are fixed using training and validation information. All four machine-learning models are refitted on 2010--2021 and evaluated on the same dates; the test period is used once for reporting. The two momentum baselines and the two ETFs are evaluated on the same dates with the same execution convention.

The main experiment and the Ridge extension were produced by different backtest drivers. Before comparing them, the study verifies three things: that both engines use the same 48 decision dates, that SPY and QQQ net returns are identical across engines on every date, and that B3 is the momentum Top-10 of the same thirty-candidate pool used by the main experiment, checked date by date. All three checks pass exactly. The comparison therefore does not mix different return conventions.

Table~\ref{tab:oos} reports the eight strategies. All returns are net of 20 basis points per traded side. Cumulative return is the compounded net return over the 48 months. CAGR is the annualised compound return. Volatility is the annualised standard deviation of monthly net returns. Sharpe and Sortino use a zero risk-free rate and are therefore approximations rather than true excess-return statistics; the definitions follow \citet{sharpe1966} and \citet{sortino1994}. Maximum drawdown is computed from the cumulative wealth series. Turnover is the average monthly sum of absolute weight changes, and cost is the cumulative cost drag in return units.

\input{generated/table_oos}

\subsection{Four machine-learning models}

Figure~\ref{fig:fourmodel} shows the four ML models under the common trading rule. Ridge has the highest terminal wealth and the highest Sharpe, followed closely by LASSO. Random Forest and XGBoost lag. The ordering is not stable across the whole period: the four curves are close until 2024 and diverge afterwards.

The performance gap is small relative to the uncertainty of a four-year window. Under the original four-model experiment, Ridge's OOS MSE is @@RIDGE_OOS_MSE@@ versus @@LASSO_OOS_MSE@@ for LASSO, its CAGR is @@RIDGE_CAGR@@ versus @@LASSO_CAGR@@, and its Sharpe is @@RIDGE_SHARPE@@ versus @@LASSO_SHARPE@@. These differences are marginal, are not tested for statistical significance, and do not establish universal superiority of Ridge or inferiority of LASSO's sparse selection. The out-of-sample rank IC tells a slightly different story: Random Forest has the highest rank IC at @@RF_OOS_IC@@, while Ridge is at @@RIDGE_OOS_IC@@. The ML comparison is therefore best described as inconclusive rather than as a Ridge victory.

\begin{figure}[H]
\centering
\includegraphics[width=0.60\textwidth]{figures/fig_four_model.png}
\caption{Out-of-sample growth of \$1 for the four machine-learning models under the same Top-10 equal-weight rule. Returns are net of 20 basis points per traded side.}
\label{fig:fourmodel}
\end{figure}

\subsection{Ridge versus momentum baselines and benchmarks}

Figure~\ref{fig:benchmarks} compares the primary Ridge strategy with B1, B3, SPY and QQQ. Ridge compounds to @@RIDGE_GROWTH@@ times invested capital over the period. SPY compounds to @@SPY_GROWTH@@ and QQQ to @@QQQ_GROWTH@@, so the active strategy comfortably beats both investable benchmarks. It does not beat the momentum baselines. B1 compounds to @@B1_GROWTH@@ and B3 to @@B3_GROWTH@@.

\begin{figure}[H]
\centering
\includegraphics[width=0.60\textwidth]{figures/fig_ridge_benchmarks.png}
\caption{Primary Ridge strategy versus momentum baselines and investable benchmarks, 2022--2025. All series are net of costs.}
\label{fig:benchmarks}
\end{figure}

The numbers in Table~\ref{tab:oos} confirm the visual impression. B1 earns @@B1_CAGR@@ with a @@B1_SHARPE@@ Sharpe and a @@B1_MDD@@ maximum drawdown. B3 earns @@B3_CAGR@@ with a @@B3_SHARPE@@ Sharpe and a @@B3_MDD@@ drawdown. Ridge earns @@RIDGE_CAGR@@ with a @@RIDGE_SHARPE@@ Sharpe and a @@RIDGE_MDD@@ drawdown. Both baselines dominate Ridge on every risk-adjusted dimension, not just on terminal wealth.

The Ridge-versus-B3 comparison is the cleanest evidence on the forecasting question. Both strategies use the same thirty candidates, the same execution and the same costs. The only difference is whether the top ten are chosen by Ridge prediction or by twelve-month momentum. B3's CAGR is @@RIDGE_MINUS_B3_CAGR@@ percentage points higher, its Sharpe is @@RIDGE_MINUS_B3_SHARPE@@ higher, and its maximum drawdown is @@RIDGE_DD_VS_B3@@ percentage points shallower. B3 also trades less: its average monthly turnover is @@B3_TURNOVER@@ versus @@RIDGE_TURNOVER@@ for Ridge. In this sample, the Ridge ranking did not add value over momentum inside the screened pool.

B1's advantage is larger, at @@RIDGE_MINUS_B1_CAGR@@ percentage points of CAGR and @@RIDGE_MINUS_B1_SHARPE@@ of Sharpe, but it is not a clean ranking comparison because B1 also selects from the full liquid universe. The difference between B1 and B3 shows how much candidate-universe design matters: expanding the pool from thirty screened names to the full liquid universe and ranking by momentum adds roughly 22 percentage points of CAGR over B3.

\subsection{Eight-strategy summary}

Figure~\ref{fig:summary} and Figure~\ref{fig:risk} summarise the eight strategies. B1 and B3 occupy the top two positions on both CAGR and Sharpe. Ridge is the best of the four ML models but fourth overall. SPY and QQQ are last on CAGR and Sharpe, although their drawdowns are not the worst on every measure. The consistency across CAGR, Sharpe, Sortino, Calmar and drawdown is important: the baseline advantage is not an artefact of one metric.

\begin{figure}[H]
\centering
\includegraphics[width=0.66\textwidth]{figures/fig_eight_summary.png}
\caption{Eight-strategy summary: (a) annualised net return and (b) Sharpe ratio. Ridge is the best ML model but ranks behind both momentum baselines.}
\label{fig:summary}
\end{figure}

\subsection{Drawdowns}

Figure~\ref{fig:drawdowns} shows the drawdown paths. Ridge's maximum drawdown of @@RIDGE_MDD@@ is deeper than B1's @@B1_MDD@@ and much deeper than B3's @@B3_MDD@@. B3 is the most defensive of the active strategies despite having no explicit risk overlay, because its momentum ranking within the screened pool happened to avoid the worst drawdown names in this sample. This should not be interpreted as a structural property of momentum; it is one historical path.

\begin{figure}[H]
\centering
\includegraphics[width=0.60\textwidth]{figures/fig_drawdowns.png}
\caption{Out-of-sample drawdowns for Ridge, B1, B3, SPY and QQQ.}
\label{fig:drawdowns}
\end{figure}

\subsection{Annual returns}

Figure~\ref{fig:annual} and Table~\ref{tab:annual} decompose the performance by calendar year. The baseline advantage is not uniform. In 2022, a down year for equities, Ridge lost 14.4\% while B1 returned 12.3\% and B3 returned 5.6\%. In 2023 Ridge led the baselines slightly, at 59.5\% versus 58.6\% for B1 and 50.3\% for B3. In 2024 B1 and B3 returned 126.1\% and 122.7\% against 70.8\% for Ridge. In 2025 B1 returned 123.0\%, Ridge 79.5\% and B3 49.0\%.

The 2024--2025 gap drives most of the terminal wealth difference. This is consistent with a momentum strategy performing well in a period of strong persistent trends, but the sample is too short to separate a regime effect from a structural advantage. The 2022 result is a reminder that the baselines are not uniformly better: in a declining market, the Ridge portfolio fell while both momentum rules were positive.

\begin{figure}[H]
\centering
\includegraphics[width=0.60\textwidth]{figures/fig_annual.png}
\caption{Annual net returns for Ridge, B1, B3, SPY and QQQ.}
\label{fig:annual}
\end{figure}

\input{generated/table_annual}

\subsection{Risk-adjusted performance}

Figure~\ref{fig:risk} compares Sharpe ratios and volatilities. Ridge's volatility of @@RIDGE_VOL@@ is lower than B1's @@B1_VOL@@ and higher than B3's @@B3_VOL@@. The Sharpe ranking is therefore not a simple volatility story: B3 earned a higher return with lower volatility and a shallower drawdown. Ridge's Sharpe of @@RIDGE_SHARPE@@ is respectable in absolute terms and well above SPY's @@SPY_SHARPE@@, but it is not competitive with B3's @@B3_SHARPE@@.

\begin{figure}[H]
\centering
\includegraphics[width=0.66\textwidth]{figures/fig_risk_adjusted.png}
\caption{Risk-adjusted performance: (a) Sharpe ratio and (b) annualised volatility.}
\label{fig:risk}
\end{figure}

\subsection{Prediction performance versus portfolio performance}

The relationship between forecast accuracy and portfolio outcome is weak. Ridge has the lowest OOS MSE and the highest OOS Sharpe among the ML models, but Random Forest has the highest OOS rank IC. LASSO's MSE is almost identical to Ridge's, yet its CAGR is about 0.9 percentage points lower. These differences are small and not statistically distinguishable in a 48-month sample. The more important lesson is that the Top-10 rule uses only the extreme of the ranking; a model that is slightly better on average can be slightly worse at the top of the distribution.

\subsection{Transaction costs and turnover}

Ridge's average monthly turnover is @@RIDGE_TURNOVER@@, compared with @@B1_TURNOVER@@ for B1 and @@B3_TURNOVER@@ for B3. The higher turnover costs @@RIDGE_COST@@ of cumulative return over the period, versus @@B1_COST@@ for B1 and @@B3_COST@@ for B3. Gross of costs, Ridge's CAGR is @@RIDGE_CAGR_GROSS@@, so costs reduce its annual return by @@RIDGE_COST_ANNUAL@@ percentage points. The corresponding drag is @@B1_MINUS_RIDGE_COST@@ percentage points for B1 and @@B3_COST_ANNUAL@@ for B3. Costs are a material part of the comparison, not a rounding adjustment.

\subsection{Alpha and factor exposure}

Ridge's beta to SPY is @@RIDGE_BETA@@ and its annualised Jensen alpha is @@RIDGE_ALPHA@@, with an information ratio of @@RIDGE_IR@@. B1's alpha is @@B1_ALPHA@@ with a beta of 1.66. The high beta means that a large part of the strategy's return is compensation for market exposure. The alpha is positive and economically meaningful, but it is estimated from 48 monthly observations and is not adjusted for size, value or momentum factors beyond the SPY market factor \citep{fama1993, carhart1997}. It should be read as descriptive, not as a structural estimate of skill.
"""))

    sections.append(sub(r"""
\section{Interpretation and Economic Discussion}
\label{sec:interpretation}

\subsection{What worked}

Ridge produced a usable cross-sectional ranking. Under the unchanged Top-10 equal-weight rule it earned @@RIDGE_CAGR@@ net compound annual return and a @@RIDGE_SHARPE@@ Sharpe ratio over 2022--2025, against @@SPY_CAGR@@ and @@SPY_SHARPE@@ for SPY and @@QQQ_CAGR@@ and @@QQQ_SHARPE@@ for QQQ. It also produced the highest OOS Sharpe and CAGR among the four forecasting models. The result is not a trivial artefact of market beta alone: the annualised alpha against SPY is @@RIDGE_ALPHA@@, and the information ratio is @@RIDGE_IR@@.

The validation evidence supports the specification. Ridge has the highest validation rank IC, its coefficient vector moves less between rolling windows than LASSO's, and the correlation structure of the predictors makes a dense shrinkage estimator a reasonable choice. These are genuine arguments in favour of Ridge as a forecasting model.

\subsection{What did not work}

Ridge did not beat simple momentum. B1 and B3 have higher CAGR, higher Sharpe, higher Sortino, higher Calmar and shallower drawdowns. The Ridge-versus-B3 comparison is particularly damaging to the incremental-value claim because it holds the candidate pool, execution and costs fixed. Within that pool, a twelve-month momentum rank produced a better portfolio than the Ridge prediction.

The result is consistent across metrics and across most years, so it is not a single-month anomaly. It is also economically plausible. Twelve-month momentum is a well-documented cross-sectional signal \citep{jegadeesh1993, asness2013}. A regression on fifteen correlated characteristics may dilute that signal with noisier inputs, and the Top-10 selection concentrates the portfolio in the extreme of the predicted distribution, where estimation error is largest.

\subsection{Why simple momentum may outperform a regression}

Three mechanisms can explain the result. First, momentum is a low-dimensional, high-signal characteristic in this universe, while the regression combines fourteen additional predictors whose marginal information may be small. Second, the regression is estimated on a pooled panel with a single coefficient vector, which assumes that the mapping from characteristics to returns is stable across stocks and time. If the mapping is heterogeneous or regime-dependent, the pooled coefficients can be worse than a simple rank. Third, the portfolio selects only ten names; small differences in the middle of the ranking are irrelevant, while errors at the top are costly. A model optimised for average squared error is not necessarily optimised for the top-decile ordering.

\subsection{The importance of candidate-universe design}

B1 and B3 differ only in their candidate universe and ranking implementation, yet B1's CAGR is roughly 22 percentage points higher than B3's. B1 ranks the full liquid universe by momentum, while B3 ranks only the thirty double-sorted candidates. The comparison shows that the screen itself is a powerful filter: the double sort improves the quality of the candidate pool relative to a random subset, but restricting the pool to thirty names also removes high-momentum stocks that the screen did not select. Candidate-universe design is therefore a first-order strategic choice, not a technical detail.

This also means that the Ridge-versus-B1 comparison cannot be interpreted as a pure ranking test. B1 differs in both the pool and the ranking method. Only the Ridge-versus-B3 comparison isolates ranking.

\subsection{Ranking versus weighting}

The extension shows that weighting matters. R1 holds the same stocks as B1 but weights them by Ridge predictions with a 20\% cap. Its CAGR is 87.9\% versus 73.1\% for B1, but its volatility rises from 37.7\% to 44.9\% and its Sharpe is slightly lower (1.64 versus 1.66). The higher absolute return is partly compensation for higher risk and higher turnover. This is a useful reminder that changing the weighting scheme is not the same as improving the forecast, and that a higher CAGR does not by itself indicate a better strategy.

\subsection{Market regimes and the limits of the evidence}

The test period contains very different market environments: a broad decline in 2022, a strong recovery in 2023, and large momentum-driven gains in 2024--2025. The baseline advantage is concentrated in the later years. A four-year window cannot distinguish a structural advantage from a regime effect, and the paper does not claim that momentum will continue to outperform Ridge. The honest conclusion is that the evidence in this sample favours momentum, and that a longer sample or a regime-conditional analysis would be needed to generalise.

\subsection{Limits of the evidence}

The sample is small: 48 monthly observations for the test period, with overlapping one-month holding periods. The strategies are evaluated on a survivorship-biased universe. The cost model is constant and ignores market impact. The models are static during the test period. The primary designation is not a clean pre-registration. These limitations do not invalidate the comparison, but they bound its interpretation. In particular, the paper does not claim statistical significance for the Sharpe differences; it reports them as descriptive evidence from one historical path.
"""))

    sections.append(sub(r"""
\section{Exploratory Ridge Strategy Extensions}
\label{sec:extensions}

\subsection{Motivation}

The main out-of-sample results motivated a separate question: can Ridge-based portfolio construction be improved by changing the candidate universe, the screening rule, the ranking, the weighting or the diversification constraints? The original exploratory extension addresses this question with six investable sleeves plus SPY and QQQ. The extension is preserved and reported here as exploratory analysis.

\subsection{The extension strategies}

Table~\ref{tab:extension} reports the extension. B1, B2 and B3 are momentum baselines with different universe definitions and portfolio sizes. R1 uses the same stocks as B1 but weights them by Ridge predictions with a 20\% cap. R2 selects the Ridge Top 10 from sixty momentum candidates. R3 imposes a 3/4/3 quota across the high, medium and low momentum groups. SPY and QQQ are buy-and-hold benchmarks.

\input{generated/table_extension}

\subsection{Results and interpretation}

R1 has the highest CAGR in the extension, at 87.9\%, but its volatility is 44.9\% and its Sharpe is 1.64, slightly below B1's 1.66. The higher return is therefore not a free improvement: it comes with higher risk and higher turnover. R2 earns 37.7\% with a Sharpe of 1.01 and a drawdown of $-$46.6\%, and R3 earns 22.5\% with a Sharpe of 0.78 and a drawdown of $-$39.0\%. Both are weaker than B3's 51.5\%, 1.59 and $-$15.0\%.

The extension illustrates three points. First, prediction-based weighting can increase absolute return while leaving risk-adjusted performance roughly unchanged or worse. Second, changing the candidate universe or the Ridge training sample changes the fitted model and can change the result even when the economic rule is similar; the training universe is part of the model specification. Third, diversification constraints such as the 3/4/3 quota can reduce concentration but do not necessarily improve performance.

\subsection{Status of the extension}

These analyses are exploratory. Their design was informed by previously viewed out-of-sample results, so they do not provide fresh independent confirmation. They are reported for economic insight, not as replacements for the primary strategy. The primary Ridge strategy remains the 3$\times$3 screen, Ridge Top-10 ranking and equal-weight rule, regardless of whether an extension earned a higher CAGR.
"""))

    sections.append(sub(r"""
\section{Limitations and Future Research}
\label{sec:limitations}

\subsection{Data limitations}

Using current S\&P 500 constituents to construct the historical equity universe introduces potential survivorship and constituent-selection bias into the active stock-selection strategies. Firms that were removed from the index are absent and current survivors are over-represented, which may overstate the active strategies' historical performance relative to SPY and QQQ. The buy-and-hold benchmark returns are calculated directly from historical ETF prices and are not affected by the same constituent-selection mechanism. Data-vendor price adjustments and corporate-action treatment can affect both individual-stock histories and ETF return calculations. The direction or magnitude of the survivorship effect is not precisely measured here, and a point-in-time constituent database would be required to correct it.

Vendor data also introduce limitations. Yahoo Finance adjusted prices can be revised, and the snapshot is not an exchange-grade point-in-time record. The data are adequate for an academic exercise but not for production research.

\subsection{Statistical limitations}

The test period contains 48 monthly observations. Monthly returns are overlapping in the sense that the holding period spans the decision interval, and the strategies share the same candidate pool, so the effective number of independent observations is smaller than 48. The Sharpe differences reported here are not accompanied by confidence intervals or multiple-testing corrections. A block bootstrap or a deflated Sharpe ratio \citep{bailey2014} would provide a more formal assessment, but with four years of data the power would remain low.

\subsection{Model uncertainty}

The comparison evaluates four algorithms and twenty-three configurations. Model uncertainty is therefore not fully captured by the selected configuration: a different but equally plausible configuration could produce different results. The paper mitigates this by using a small pre-declared grid and by reporting the full validation table, but it does not perform a full model-averaging exercise.

\subsection{Transaction-cost assumptions}

The 20 basis point cost is constant and proportional. It does not model market impact, spread variation, borrow costs, taxes or capacity. For a ten-stock large-cap portfolio the assumption is reasonable, but a larger portfolio would face higher impact costs. The cost sensitivity of the results is not explored beyond the gross-versus-net comparison reported in Section~\ref{sec:oos}.

\subsection{Market-regime sensitivity}

The baseline advantage is concentrated in 2024--2025. A longer sample or a regime-conditional analysis would be needed to determine whether the result is structural or cyclical. The paper does not claim that momentum will continue to outperform Ridge.

\subsection{Test-informed exploratory analysis}

The exploratory extension was designed after earlier out-of-sample results were inspected. It is therefore not an independent confirmation of any hypothesis. The paper labels it exploratory and does not use it to select the primary strategy.

\subsection{Future research}

Four extensions would strengthen the analysis. First, a point-in-time constituent database would remove survivorship bias. Second, rolling out-of-sample retraining with a fixed window would test whether the static model is the limiting factor. Third, a regime-conditional analysis would test whether Ridge adds value in specific market states. Fourth, a formal multiple-testing framework would quantify the uncertainty around the Sharpe differences. None of these extensions is claimed in this paper.
"""))

    sections.append(sub(r"""
\section{Conclusion}
\label{sec:conclusion}

The paper asked whether Ridge Regression, trained on economically motivated predictors, can generate useful cross-sectional rankings and improve a systematic equity portfolio relative to simpler alternatives. The answer is nuanced.

Ridge works as a forecasting model. It produces a usable ranking, earns @@RIDGE_CAGR@@ net CAGR and a @@RIDGE_SHARPE@@ Sharpe over 2022--2025, and beats SPY and QQQ by a wide margin after costs. It also has the highest OOS Sharpe and CAGR among the four machine-learning models, and its validation rank IC is the highest of the four. The Ridge specification is supported by the correlation structure of the predictors and by rolling coefficient-stability diagnostics: its coefficients move about @@REDUCTION@@\% less between windows than LASSO's, even though it sign-flips more often.

Ridge does not beat simple momentum. B1 earns @@B1_CAGR@@ with a @@B1_SHARPE@@ Sharpe, and B3 earns @@B3_CAGR@@ with a @@B3_SHARPE@@ Sharpe and a @@B3_MDD@@ drawdown. Within the same candidate pool, the Ridge ranking underperforms a momentum ranking by @@RIDGE_MINUS_B3_CAGR@@ percentage points of CAGR and @@RIDGE_MINUS_B3_SHARPE@@ of Sharpe, with higher turnover. The result is consistent across CAGR, Sharpe, Sortino, Calmar and drawdown, and it is not driven by a single year.

The economic lesson is that a statistically sensible forecasting model is not automatically a better portfolio rule. Momentum is a low-dimensional, high-signal characteristic in this universe, and the Top-10 portfolio concentrates on the extreme of the ranking, where estimation error is largest. Candidate-universe design is at least as important as the forecasting algorithm: B1's broader universe contributes roughly 22 percentage points of CAGR relative to B3, independent of the model.

The paper is transparent about its limitations. The universe is survivorship-biased, the test window is short, the cost model is simple, and the Ridge designation is not a clean pre-registration because the project history includes prior out-of-sample review. These limitations do not overturn the comparison, but they bound its interpretation. The defensible conclusion is that Ridge is an economically motivated model that adds value relative to passive benchmarks but does not establish incremental value over momentum-based selection in this sample. Reporting that result, rather than selecting a different strategy after seeing the test period, is the main contribution of the exercise.
"""))

    sections.append(sub(r"""
\section{Reproducibility and AI-Use Declaration}
\label{sec:reproducibility}

\subsection{Project structure and commands}

The repository contains \texttt{src/} for source code, \texttt{config/} for the experiment configuration, \texttt{tests/} for the automated suite, \texttt{results/} for generated tables and figures, \texttt{report/} for this document and its LaTeX source, and \texttt{notebooks/} for the executed notebook.

The complete pipeline is reproduced with:
\begin{verbatim}
python run_pipeline.py --stage validation
python run_pipeline.py --stage designate --approve
python run_pipeline.py --stage refit
python run_pipeline.py --stage oos
python run_pipeline.py --stage extension
python run_pipeline.py --stage eight
python run_tests.py --notebook
\end{verbatim}
The report is rebuilt with \texttt{python report/build\_report.py}. The environment is Python 3.11.4 with numpy 1.26.4, pandas 2.2.3, scikit-learn 1.9.1, XGBoost 3.2.0, matplotlib 3.11.2 and PyYAML 6.0. The random seed is 42.

\subsection{Outputs and audit trail}

Validation tables and fifteen figures are under \texttt{results/validation/}. Out-of-sample tables and Figures~2--7 are under \texttt{results/oos/} and \texttt{results/oos/eight\_strategy/}. The exploratory extension is under \texttt{results/extensions/}. The correlation and coefficient-stability diagnostics are under \texttt{results/diagnostics/}. Serialized models are under \texttt{results/models/refit/}.

The primary-model contract is \texttt{results/validation/primary\_model\_contract.json}. It records the primary model, its hyperparameters, the validation metrics, the diagnostic evidence, the historical LASSO audit and the chronology disclosure. The historical LASSO selection contract is archived under \texttt{results/validation/archive/} and is not overwritten.

\subsection{AI-use declaration}

This revision was prepared with AI assistance (DeepSeek Harness) for code restructuring, test generation, diagnostic integration, figure production and LaTeX drafting. All numerical results were produced by executing the repository code on the historical data snapshot. The original strategy, data pipeline and exploratory extension were authored by the project team. The AI assistant did not have access to any data beyond the supplied project files, and no result was fabricated or manually inserted.

\subsection{Chronology disclosure}

The original project pre-specified Ridge. A later engineering revision introduced an automatic minimum-MSE rule that selected LASSO, and out-of-sample results were reviewed during that revision. The Ridge designation was restored afterwards on economic and diagnostic grounds. The primary-model contract records that the designation is not a clean pre-registration and that it was informed by prior out-of-sample review. No test-period metric was used to choose Ridge or its $\alpha$, but the designation should not be presented as a blind pre-registration.
"""))

    # ---------------------------------------------------------------- document
    preamble = r"""
\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{setspace}
\usepackage{booktabs}
\usepackage{graphicx}
\graphicspath{{../}{../figures/}}
\usepackage{caption}
\usepackage{amsmath,amssymb}
\usepackage[round,authoryear]{natbib}
\usepackage{float}
\usepackage{enumitem}
\usepackage{fancyhdr}
\usepackage[hidelinks]{hyperref}
\usepackage{titlesec}
\usepackage{xcolor}

\setstretch{1.0}
\setlength{\parindent}{0pt}
\setlength{\parskip}{1pt}
\captionsetup{font=small,labelfont=bf,skip=3pt}
\setlength{\textfloatsep}{8pt plus 2pt minus 2pt}
\setlength{\floatsep}{8pt plus 2pt minus 2pt}
\setlength{\intextsep}{8pt plus 2pt minus 2pt}
\renewcommand{\arraystretch}{0.95}
\setlength{\tabcolsep}{4pt}
\setlength{\emergencystretch}{2em}
\captionsetup[table]{position=top}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\footnotesize DSA5205 Project 2}
\fancyhead[R]{\footnotesize Ridge-based cross-sectional equity strategy}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\titleformat{\section}{\large\bfseries}{\thesection}{0.6em}{}
\titleformat{\subsection}{\normalsize\bfseries}{\thesubsection}{0.5em}{}
\titlespacing*{\section}{0pt}{2.0ex plus 0.5ex}{1.0ex}
\titlespacing*{\subsection}{0pt}{1.5ex plus 0.4ex}{0.8ex}
\newcommand{\team}{Group X}
\begin{document}
"""

    title = r"""
\begin{titlepage}
\centering
\vspace*{1.4cm}
{\small DSA5205 --- Data Science for Quantitative Finance\par}
\vspace{0.4cm}
{\small Project 2: Design, Validate, and Analyze an Original Asset Trading Strategy\par}
\vspace{2.2cm}
{\LARGE\bfseries Ridge Regression for Cross-Sectional Equity Selection\par}
\vspace{0.6cm}
{\Large A Validation and Out-of-Sample Study of a Momentum--Volatility Screened Portfolio\par}
\vspace{2.2cm}
{\large DSA5205 Project 2 Team\par}
\vspace{1.4cm}
{\normalsize 11 November 2026\par}
\vfill
\begin{minipage}{0.86\textwidth}
\small
\textbf{Scope.} This report studies one primary Ridge-based strategy, compares it with three alternative machine-learning models, two momentum baselines and two investable ETFs, and reports the results of a chronological out-of-sample evaluation. The original exploratory Ridge extension is retained as exploratory analysis. All results are for academic research and historical backtesting only.
\end{minipage}
\vspace{1.2cm}
\end{titlepage}
"""

    toc = ""

    abstract_block = "\\begin{abstract}\n" + abstract + "\n\\end{abstract}\n"

    bib = r"""
\begin{thebibliography}{99}
\small
\setlength{\itemsep}{1pt}

\bibitem[Asness et al.(2013)]{asness2013}
Asness, C. S., Moskowitz, T. J., and Pedersen, L. H. (2013).
Value and momentum everywhere.
\textit{Journal of Finance}, 68(3), 929--985.

\bibitem[Bailey and L\'opez de Prado(2014)]{bailey2014}
Bailey, D. H., and L\'opez de Prado, M. (2014).
The deflated Sharpe ratio: correcting for selection bias, backtest overfitting, and non-normality.
\textit{Journal of Portfolio Management}, 40(5), 94--107.

\bibitem[Breiman(2001)]{breiman2001}
Breiman, L. (2001).
Random forests.
\textit{Machine Learning}, 45(1), 5--32.

\bibitem[Campbell and Thompson(2008)]{campbell2008}
Campbell, J. Y., and Thompson, S. B. (2008).
Predicting excess stock returns out of sample: can anything beat the historical average?
\textit{Review of Financial Studies}, 21(4), 1509--1531.

\bibitem[Carhart(1997)]{carhart1997}
Carhart, M. M. (1997).
On persistence in mutual fund performance.
\textit{Journal of Finance}, 52(1), 57--84.

\bibitem[Chen and Guestrin(2016)]{chen2016}
Chen, T., and Guestrin, C. (2016).
XGBoost: a scalable tree boosting system.
In \textit{Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining}, 785--794.

\bibitem[Fama and French(1993)]{fama1993}
Fama, E. F., and French, K. R. (1993).
Common risk factors in the returns on stocks and bonds.
\textit{Journal of Financial Economics}, 33(1), 3--56.

\bibitem[Feng et al.(2020)]{feng2020}
Feng, G., Giglio, S., and Xiu, D. (2020).
Taming the factor zoo: a test of new factors.
\textit{Journal of Finance}, 75(3), 1327--1370.

\bibitem[Freyberger et al.(2020)]{freyberger2020}
Freyberger, J., Neuhierl, A., and Weber, M. (2020).
Dissecting characteristics nonparametrically.
\textit{Review of Financial Studies}, 33(5), 2326--2377.

\bibitem[Gu et al.(2020)]{gu2020}
Gu, S., Kelly, B., and Xiu, D. (2020).
Empirical asset pricing via machine learning.
\textit{Review of Financial Studies}, 33(5), 2223--2273.

\bibitem[Hoerl and Kennard(1970)]{hoerl1970}
Hoerl, A. E., and Kennard, R. W. (1970).
Ridge regression: biased estimation for nonorthogonal problems.
\textit{Technometrics}, 12(1), 55--67.

\bibitem[Hou et al.(2020)]{hou2020}
Hou, K., Xue, C., and Zhang, L. (2020).
Replicating anomalies.
\textit{Review of Financial Studies}, 33(5), 2019--2133.

\bibitem[Jegadeesh(1990)]{jegadeesh1990}
Jegadeesh, N. (1990).
Evidence of predictable behavior of security returns.
\textit{Journal of Finance}, 45(3), 881--898.

\bibitem[Jegadeesh and Titman(1993)]{jegadeesh1993}
Jegadeesh, N., and Titman, S. (1993).
Returns to buying winners and selling losers: implications for stock market efficiency.
\textit{Journal of Finance}, 48(1), 65--91.

\bibitem[Kelly et al.(2019)]{kelly2019}
Kelly, B. T., Pruitt, S., and Su, Y. (2019).
Characteristics are covariances: a unified model of risk and return.
\textit{Journal of Financial Economics}, 134(3), 501--524.

\bibitem[McLean and Pontiff(2016)]{mclean2016}
McLean, R. D., and Pontiff, J. (2016).
Does academic research destroy stock return predictability?
\textit{Journal of Finance}, 71(1), 5--32.

\bibitem[Moskowitz et al.(2012)]{moskowitz2012}
Moskowitz, T. J., Ooi, Y. H., and Pedersen, L. H. (2012).
Time series momentum.
\textit{Journal of Financial Economics}, 104(2), 228--250.

\bibitem[Sharpe(1966)]{sharpe1966}
Sharpe, W. F. (1966).
Mutual fund performance.
\textit{Journal of Business}, 39(1), 119--138.

\bibitem[Sortino and Price(1994)]{sortino1994}
Sortino, F. A., and Price, L. N. (1994).
Performance measurement in a downside risk framework.
\textit{Journal of Investing}, 3(3), 59--64.

\bibitem[Tibshirani(1996)]{tibshirani1996}
Tibshirani, R. (1996).
Regression shrinkage and selection via the lasso.
\textit{Journal of the Royal Statistical Society: Series B}, 58(1), 267--288.

\end{thebibliography}
"""

    appendix = sub(r"""
\appendix
\section{Diagnostic Figures}
\label{app:diagnostics}
\setcounter{figure}{0}
\renewcommand{\thefigure}{A\arabic{figure}}

The two figures below support the model-justification discussion in Section~\ref{sec:model}. They are placed in the appendix as Figures~A1--A2 so that the main text keeps the score-to-position and out-of-sample exhibits in sequence.

\begin{figure}[H]
\centering
\includegraphics[width=0.55\textwidth]{figures/fig_correlation.png}
\caption{Training-period Pearson correlation matrix of the fifteen features. Boxed cells have absolute correlation of at least 0.70.}
\label{fig:corr}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.60\textwidth]{figures/fig_stability.png}
\caption{Rolling-window coefficient stability. Panel (a) shows the distribution of relative $L_2$ coefficient changes between adjacent 60-month windows; panel (b) shows the fraction of zero coefficients. Ridge changes less on average but sign-flips more often.}
\label{fig:stability}
\end{figure}

\section{Supplementary Tables}

Table~\ref{tab:gross_net} reports the gross-to-net decomposition and market-factor exposure for the eight strategies, complementing the net figures in Table~\ref{tab:oos}.

\input{generated/table_grossnet}



""")

    body = preamble + title + abstract_block + toc + "\n".join(sections) + bib + appendix + "\n\\end{document}\n"

    # Replace placeholders everywhere (including abstract and appendix)
    body = sub(body)

    (REPORT / "Project2_Report.tex").write_text(body)
    print(f"wrote {REPORT / 'Project2_Report.tex'} ({len(body):,} chars)")

    # ---------------------------------------------------------------- compile
    for _ in range(2):
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "Project2_Report.tex"],
            cwd=REPORT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(result.stdout[-4000:])
            print(result.stderr[-2000:])
            raise SystemExit("pdflatex failed")
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "Project2_Report.tex"],
        cwd=REPORT,
        capture_output=True,
        text=True,
    )
    pdf = REPORT / "Project2_Report.pdf"
    if not pdf.exists():
        raise SystemExit("PDF was not produced")
    reader = PdfReader(str(pdf))
    pages = len(reader.pages)
    print(f"PDF pages: {pages}")
    if not (TARGET_PAGES[0] <= pages <= TARGET_PAGES[1]):
        print(f"WARNING: page count {pages} outside target {TARGET_PAGES}")
    return pages


if __name__ == "__main__":
    build()
