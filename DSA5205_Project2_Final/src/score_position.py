"""Score-to-position illustration for the primary Ridge strategy.

The figure uses one real decision date from the saved out-of-sample
predictions.  It shows the thirty candidate scores, the Top-10 cut-off and the
resulting equal weights.  No illustrative or synthetic scores are used.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from .data_pipeline import ensure_output_dirs, load_config, resolve
from .utils import utc_now, write_json

DEFAULT_DATE = "2025-06-30"


def build_score_position_table(cfg: Optional[Mapping[str, Any]] = None, decision_date: str = DEFAULT_DATE) -> pd.DataFrame:
    cfg = load_config() if cfg is None else cfg
    path = resolve(cfg["paths"]["oos"]) / "oos_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing OOS predictions: {path}")
    frame = pd.read_csv(path, parse_dates=["decision_date"])
    block = frame.loc[(frame.model == "ridge") & (frame.decision_date == pd.Timestamp(decision_date))].copy()
    if len(block) != 30:
        raise ValueError(f"Expected 30 candidates on {decision_date}, found {len(block)}")
    block = block.sort_values(["prediction", "ticker"], ascending=[False, True]).reset_index(drop=True)
    block["rank"] = np.arange(1, len(block) + 1)
    block["selected"] = block["rank"] <= 10
    block["target_weight"] = np.where(block["selected"], 0.1, 0.0)
    if not np.isclose(block["target_weight"].sum(), 1.0):
        raise AssertionError("Target weights must sum to one")
    if int(block["selected"].sum()) != 10:
        raise AssertionError("Exactly ten stocks must be selected")
    return block


def run_score_position_example(
    cfg: Optional[Mapping[str, Any]] = None,
    decision_date: str = DEFAULT_DATE,
) -> Dict[str, Any]:
    cfg = load_config() if cfg is None else cfg
    ensure_output_dirs(cfg)
    table = build_score_position_table(cfg, decision_date)
    out_dir = resolve(cfg["paths"]["oos"])
    fig_dir = resolve(cfg["paths"]["oos_figures"])
    csv_path = out_dir / "score_to_position_example.csv"
    table[
        [
            "decision_date",
            "ticker",
            "prediction",
            "rank",
            "selected",
            "target_weight",
            "mom_2_12",
            "target_forward_return",
            "execution_date",
            "label_end_date",
        ]
    ].to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    colors = np.where(table["selected"], "#12355b", "#c7ccd4")
    bars = ax.bar(table["rank"], table["prediction"], color=colors, width=0.78, edgecolor="none")
    ax.axvline(10.5, color="#b3261e", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Candidate rank by Ridge predicted return", fontsize=9.5)
    ax.set_ylabel("Predicted next-period return", fontsize=9.5)
    ax.set_title(f"Score-to-position example: Ridge predictions on {pd.Timestamp(decision_date).date()}", fontsize=10.5)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(axis="y", alpha=0.25)
    ax.set_xticks(table["rank"])
    ax.set_xticklabels(table["ticker"], rotation=90, fontsize=7.5)
    ax2 = ax.twinx()
    ax2.step(table["rank"], table["target_weight"], where="mid", color="#b3261e", linewidth=1.6)
    ax2.set_ylabel("Target portfolio weight", fontsize=9.5)
    ax2.set_ylim(0, 0.12)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.text(
        0.99,
        0.97,
        "Navy = selected Top 10 (10% each)\nGrey = not held (0%)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#999999", alpha=0.9),
    )
    fig.tight_layout()
    fig_path = fig_dir / "fig_score_to_position.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    write_json(
        out_dir / "score_to_position_metadata.json",
        {
            "generated_utc": utc_now(),
            "decision_date": str(pd.Timestamp(decision_date).date()),
            "model": "ridge",
            "n_candidates": int(len(table)),
            "n_selected": int(table["selected"].sum()),
            "weight_per_selected": 0.1,
            "weight_sum": float(table["target_weight"].sum()),
            "source": str(csv_path.relative_to(resolve(cfg["paths"]["results"]))),
            "figure": str(fig_path.relative_to(resolve(cfg["paths"]["results"]))),
            "note": "Real saved OOS predictions; no illustrative scores.",
        },
    )
    return {"table": table, "csv": csv_path, "figure": fig_path, "decision_date": decision_date}


if __name__ == "__main__":  # pragma: no cover
    result = run_score_position_example()
    print(result["table"][["rank", "ticker", "prediction", "target_weight"]].head(12).to_string(index=False))
