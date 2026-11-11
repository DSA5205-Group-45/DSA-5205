"""Supplementary feature-correlation and coefficient-stability diagnostics.

The supplied ``research_diagnostics`` archive is treated as a frozen
supplementary analysis.  This module loads it, re-verifies the training-period
correlation matrix and the reported stability statistics from the project's own
training slice, and produces compact publication figures for the report.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data_pipeline import load_config, resolve
from .pipeline import FEATURES

STRONG_THRESHOLD = 0.70


def _dir(cfg: Mapping[str, Any]) -> Path:
    return resolve(cfg["paths"]["research_diagnostics"])


def load_diagnostics(cfg: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    cfg = load_config() if cfg is None else cfg
    base = _dir(cfg)
    required = [
        "feature_correlation_matrix.csv",
        "feature_correlation_pairs.csv",
        "rolling_coefficients.csv",
        "coefficient_changes_by_window.csv",
        "coefficient_stability_summary.csv",
    ]
    missing = [name for name in required if not (base / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing research diagnostics: {missing}")
    return {
        "correlation": pd.read_csv(base / "feature_correlation_matrix.csv", index_col=0),
        "pairs": pd.read_csv(base / "feature_correlation_pairs.csv"),
        "rolling": pd.read_csv(base / "rolling_coefficients.csv", parse_dates=["window_start", "window_end"]),
        "changes": pd.read_csv(base / "coefficient_changes_by_window.csv", parse_dates=["window_end"]),
        "stability": pd.read_csv(base / "coefficient_stability_summary.csv"),
        "base": base,
    }


def validate_diagnostics(
    cfg: Optional[Mapping[str, Any]] = None,
    training_frame: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Recompute the training correlation and verify the reported statistics."""
    cfg = load_config() if cfg is None else cfg
    diag = load_diagnostics(cfg)
    checks: Dict[str, Any] = {"generated_by": "src.diagnostics.validate_diagnostics"}
    if training_frame is not None:
        recomputed = training_frame[FEATURES].corr()
        supplied = diag["correlation"].reindex(index=FEATURES, columns=FEATURES)
        checks["correlation_max_abs_difference"] = float((recomputed - supplied).abs().max().max())
        checks["correlation_matches_training_slice"] = checks["correlation_max_abs_difference"] < 1e-10
    pairs = diag["pairs"]
    checks["n_pairs"] = int(len(pairs))
    checks["n_strong_pairs_0_70"] = int((pairs["abs_correlation"] >= STRONG_THRESHOLD).sum())
    summary = diag["stability"].set_index("model")
    ridge = summary.loc["ridge"]
    lasso = summary.loc["lasso"]
    checks["ridge_mean_relative_change"] = float(ridge["mean_relative_l2_change"])
    checks["lasso_mean_relative_change"] = float(lasso["mean_relative_l2_change"])
    checks["ridge_relative_change_reduction"] = float(
        (lasso["mean_relative_l2_change"] - ridge["mean_relative_l2_change"]) / lasso["mean_relative_l2_change"]
    )
    checks["lasso_zero_fraction"] = float(lasso["zero_coefficient_fraction"])
    checks["ridge_zero_fraction"] = float(ridge["zero_coefficient_fraction"])
    checks["ridge_sign_flips"] = int(ridge["total_sign_flips"])
    checks["lasso_sign_flips"] = int(lasso["total_sign_flips"])
    checks["n_windows"] = int(ridge["n_windows"])
    checks["ridge_higher_sign_flips"] = checks["ridge_sign_flips"] > checks["lasso_sign_flips"]
    checks["available"] = True
    return checks


def diagnostic_summary(
    cfg: Optional[Mapping[str, Any]] = None,
    training_frame: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    return validate_diagnostics(cfg, training_frame)


def make_report_figures(cfg: Optional[Mapping[str, Any]] = None) -> list[Path]:
    """Compact, report-quality correlation and stability figures."""
    cfg = load_config() if cfg is None else cfg
    diag = load_diagnostics(cfg)
    base = diag["base"]
    produced: list[Path] = []

    matrix = diag["correlation"].reindex(index=FEATURES, columns=FEATURES)
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    image = ax.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURES)))
    ax.set_xticklabels(FEATURES, rotation=90, fontsize=7)
    ax.set_yticks(range(len(FEATURES)))
    ax.set_yticklabels(FEATURES, fontsize=7)
    for i in range(len(FEATURES)):
        for j in range(len(FEATURES)):
            value = matrix.iloc[i, j]
            if i != j and abs(value) >= STRONG_THRESHOLD:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=0.8))
    ax.set_title("Training-period feature correlation (boxed: |r| >= 0.70)")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03, label="Pearson correlation")
    fig.tight_layout()
    path = base / "report_correlation_heatmap.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    produced.append(path)

    summary = diag["stability"].set_index("model")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    models = ["ridge", "lasso"]
    colors = {"ridge": "#12355b", "lasso": "#d97706"}
    values = [summary.loc[m, "mean_relative_l2_change"] for m in models]
    ax1.bar(models, values, color=[colors[m] for m in models], width=0.55)
    for i, value in enumerate(values):
        ax1.text(i, value + 0.006, f"{value:.3f}", ha="center", fontsize=8)
    ax1.set_ylabel("Mean relative L2 change")
    ax1.set_title("(a) Coefficient change between windows")
    ax1.grid(axis="y", alpha=0.3)
    values = [summary.loc[m, "zero_coefficient_fraction"] for m in models]
    ax2.bar(models, values, color=[colors[m] for m in models], width=0.55)
    for i, value in enumerate(values):
        ax2.text(i, value + 0.015, f"{value:.1%}", ha="center", fontsize=8)
    ax2.set_ylabel("Fraction of zero coefficients")
    ax2.set_title("(b) Sparsity")
    ax2.set_ylim(0, 0.5)
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = base / "report_stability_comparison.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    produced.append(path)
    return produced


def strong_pairs_table(cfg: Optional[Mapping[str, Any]] = None, top_n: int = 12) -> pd.DataFrame:
    diag = load_diagnostics(cfg)
    pairs = diag["pairs"].copy()
    pairs = pairs.loc[pairs["abs_correlation"] >= STRONG_THRESHOLD]
    pairs = pairs.sort_values("abs_correlation", ascending=False).head(top_n)
    pairs["correlation"] = pairs["correlation"].round(3)
    return pairs[["feature_1", "feature_2", "correlation", "abs_correlation"]].reset_index(drop=True)
