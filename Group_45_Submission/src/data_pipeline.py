"""Data access, chronological splitting and label-embargo enforcement.

Design notes
------------
* The raw data build, universe construction, feature engineering, screening
  and label generation are **not** re-implemented here.  They are provided by
  the preserved authoritative module :mod:`src.pipeline`, which is copied
  verbatim from ``DSA5205_Project2_Pipeline``.
* This module only *loads* the already-built model-ready panel, enforces the
  chronological split with a realized-label embargo, and builds a compact
  derived candidate panel for the nine-strategy Ridge extension.
* Every fitted transformation is applied downstream on the training slice
  only (see :mod:`src.models` and :mod:`src.validation`).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from .pipeline import FEATURES, ROOT, attach_labels, decision_dates
from .utils import sha256_file, write_json

REQUIRED_MODEL_COLUMNS = [
    "decision_date",
    "execution_date",
    "label_end_date",
    "ticker",
    "split",
    "target_forward_return",
    "spy_return",
    "qqq_return",
    "mom_2_12",
    "momentum_group",
    "volatility_group",
] + FEATURES


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    path = Path(path) if path is not None else ROOT / "config/experiment_config.yaml"
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["_config_path"] = str(path)
    cfg["_project_root"] = str(ROOT)
    return cfg


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def ensure_output_dirs(cfg: Mapping[str, Any]) -> None:
    for key in (
        "results",
        "validation",
        "validation_figures",
        "oos",
        "oos_figures",
        "ridge_extension",
        "ridge_extension_figures",
        "models",
        "research_diagnostics",
        "eight_strategy",
    ):
        resolve(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)


def pipeline_config(cfg: Mapping[str, Any], frequency: Optional[str] = None) -> Dict[str, Any]:
    """Adapt ``experiment_config.yaml`` to the signature expected by pipeline.py."""
    base = json.loads((ROOT / "config/pipeline.json").read_text())
    data = cfg["data"]
    base.update(
        {
            "rebalance_frequency": frequency or cfg["project"]["frequency"],
            "research_start": data["train_start"],
            "research_end": data["research_end"],
            "train_end_exclusive": data["train_end_exclusive"],
            "validation_end_exclusive": data["validation_end_exclusive"],
            "test_end_exclusive": data["test_end_exclusive"],
            "liquid_universe_size": data["liquid_universe_size"],
            "min_eligible_stocks": data["min_eligible_stocks"],
            "min_price": data["min_price"],
            "min_adv20": data["min_adv20"],
            "min_coverage": data["min_coverage"],
            "cost_bps_per_side": cfg["portfolio"]["cost_bps_per_side"],
            "random_seed": cfg["project"]["random_seed"],
        }
    )
    return base


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    for column in ("decision_date", "execution_date", "label_end_date", "date"):
        if column in df.columns:
            df[column] = pd.to_datetime(df[column])
    return df


def load_model_ready(cfg: Mapping[str, Any], frequency: Optional[str] = None) -> pd.DataFrame:
    """Load and validate the model-ready panel built by the original pipeline."""
    frequency = frequency or cfg["project"]["frequency"]
    path = resolve(cfg["data"]["model_ready_dir"]) / frequency / "model_ready.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Model-ready dataset not found: {path}. "
            "Run the original data pipeline (run_pipeline.py --frequency monthly) first, "
            "or point data.model_ready_dir at an existing build."
        )
    df = pd.read_parquet(path)
    df = _parse_dates(df)
    missing = [c for c in REQUIRED_MODEL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Model-ready dataset is missing required columns: {missing}")
    if df.empty:
        raise ValueError("Model-ready dataset is empty")
    if df.ticker.astype(str).str.startswith("SYN").any():
        raise ValueError("Synthetic tickers detected in an official data build")
    if df.duplicated(["decision_date", "ticker"]).any():
        raise ValueError("Duplicate decision-date/ticker rows")
    if not np.isfinite(df[FEATURES + ["target_forward_return"]].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite feature or target values in the model-ready dataset")
    if not (df.decision_date < df.execution_date).all():
        raise ValueError("Execution date must be strictly after the decision date")
    if not (df.execution_date < df.label_end_date).all():
        raise ValueError("Label end date must be strictly after the execution date")
    counts = df.groupby("decision_date").ticker.nunique()
    if (counts != 30).any():
        bad = counts[counts != 30].head().to_dict()
        raise ValueError(f"Every decision date must have 30 candidates; got {bad}")
    df["_frequency"] = frequency
    return df.sort_values(["decision_date", "ticker"]).reset_index(drop=True)


def chronological_splits(
    df: pd.DataFrame, cfg: Mapping[str, Any]
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Split by decision date and embargo labels that settle after the boundary.

    The rule is deliberately conservative: a row is eligible for fitting or
    tuning only when ``label_end_date <= boundary``.  This is the convention
    used by the authoritative pipeline and prevents a one-month label that
    settles in January/February of the next split from leaking backwards.
    """
    data = cfg["data"]
    train_end = pd.Timestamp(data["train_end_exclusive"])
    val_start = pd.Timestamp(data["validation_start"])
    val_end = pd.Timestamp(data["validation_end_exclusive"])
    test_start = pd.Timestamp(data["test_start"])
    test_end = pd.Timestamp(data["test_end_exclusive"])
    refit_end = pd.Timestamp(data["refit_end_exclusive"])

    d = df["decision_date"]
    l = df["label_end_date"]

    splits = {
        "train": df[(d < train_end) & (l <= train_end)].copy(),
        "validation": df[(d >= val_start) & (d < val_end) & (l <= val_end)].copy(),
        "test": df[(d >= test_start) & (d < test_end)].copy(),
        "refit": df[(d < refit_end) & (l <= refit_end)].copy(),
    }

    # Cross-check the pipeline's own split label against our decision-date rule.
    expected = np.select(
        [d < train_end, d < val_end], ["train", "validation"], default="test"
    )
    if not (expected == df["split"].to_numpy()).all():
        raise ValueError("Stored split column disagrees with the configured decision-date boundaries")
    if splits["train"].empty or splits["validation"].empty or splits["test"].empty:
        raise ValueError("One of the chronological splits is empty")
    if splits["refit"].empty:
        raise ValueError("Refit slice is empty")

    # Hard leakage guards.
    if splits["train"].label_end_date.max() > train_end:
        raise AssertionError("Training labels settle after the training cutoff")
    if splits["validation"].label_end_date.max() > val_end:
        raise AssertionError("Validation labels settle after the validation cutoff")
    if splits["refit"].label_end_date.max() > refit_end:
        raise AssertionError("Refit labels settle after the refit cutoff")
    if splits["test"].decision_date.min() < test_start:
        raise AssertionError("Test slice contains pre-test decision dates")

    summary = []
    for name in ("train", "validation", "refit", "test"):
        part = splits[name]
        summary.append(
            {
                "split": name,
                "decision_start": part.decision_date.min(),
                "decision_end": part.decision_date.max(),
                "label_end_max": part.label_end_date.max(),
                "n_decision_dates": int(part.decision_date.nunique()),
                "n_rows": int(len(part)),
                "n_tickers": int(part.ticker.nunique()),
                "label_embargo": data["label_embargo"],
            }
        )
    summary_df = pd.DataFrame(summary)

    # Document which cohorts were removed by the embargo.
    def _excluded(period_mask: pd.Series, boundary: pd.Timestamp, label: str) -> list:
        part = df[period_mask]
        dropped = part[part.label_end_date > boundary]
        return [
            {
                "split": label,
                "decision_date": str(date.date()),
                "label_end_date": str(group.label_end_date.iloc[0].date()),
                "cutoff": str(boundary.date()),
                "reason": "label settles after split boundary",
            }
            for date, group in dropped.groupby("decision_date")
        ]

    excluded = (
        _excluded(d < train_end, train_end, "train")
        + _excluded((d >= val_start) & (d < val_end), val_end, "validation")
        + _excluded(d < refit_end, refit_end, "refit")
    )
    return splits, summary_df, pd.DataFrame(excluded)


def data_quality_report(df: pd.DataFrame, splits: Mapping[str, pd.DataFrame]) -> Dict[str, Any]:
    counts = df.groupby("decision_date").ticker.nunique()
    issues = []
    if df.duplicated(["decision_date", "ticker"]).any():
        issues.append("duplicate decision_date/ticker")
    if not np.isfinite(df[FEATURES].to_numpy(dtype=float)).all():
        issues.append("non-finite features")
    if not np.isfinite(df["target_forward_return"].to_numpy(dtype=float)).all():
        issues.append("non-finite target")
    if (counts != 30).any():
        issues.append("decision dates without exactly 30 candidates")
    if not (df.decision_date < df.execution_date).all():
        issues.append("decision date not before execution date")
    if not (df.execution_date < df.label_end_date).all():
        issues.append("execution date not before label end date")
    return {
        "pass": not issues,
        "issues": issues,
        "n_rows": int(len(df)),
        "n_decision_dates": int(df.decision_date.nunique()),
        "decision_start": str(df.decision_date.min().date()),
        "decision_end": str(df.decision_date.max().date()),
        "rows_per_date_min": int(counts.min()),
        "rows_per_date_max": int(counts.max()),
        "n_features": len(FEATURES),
        "features": FEATURES,
        "target_mean": float(df.target_forward_return.mean()),
        "target_std": float(df.target_forward_return.std(ddof=1)),
        "split_rows": {k: int(len(v)) for k, v in splits.items()},
        "split_dates": {k: int(v.decision_date.nunique()) for k, v in splits.items()},
    }


# ---------------------------------------------------------------------------
# Derived candidate panel for the nine-strategy Ridge extension
# ---------------------------------------------------------------------------
def build_extension_panel(
    cfg: Mapping[str, Any],
    source_root: Path,
    frequency: Optional[str] = None,
    force: bool = False,
) -> pd.DataFrame:
    """Recreate the extension's ADV20 Top-300 candidate panel and its labels.

    This mirrors cell 3-5 of the authoritative notebook
    ``03_ridge_nine_strategy_comparison.ipynb`` exactly, but caches the result
    as a compact derived parquet so the packaged project does not need to ship
    the 331 MB full feature panel.
    """
    source_root = Path(source_root)
    panel_path = source_root / "data/processed_with_nan/full_feature_panel.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(f"Full feature panel not found: {panel_path}")
    out_path = resolve(cfg["data"]["extension_panel"])
    if out_path.exists() and not force:
        return pd.read_parquet(out_path)

    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    pcfg = pipeline_config(cfg, frequency)
    groups = {date: group for date, group in panel.groupby("date", sort=True)}

    records, audit = [], []
    for date in decision_dates(panel, pcfg):
        g = groups.get(date)
        if g is None:
            audit.append({"decision_date": date, "eligible": 0, "ok": False})
            continue
        g = g.loc[~g.ticker.isin(["SPY", "QQQ"])].copy()
        g = g.loc[
            (g.close >= pcfg["min_price"])
            & (g.adv_20 >= pcfg["min_adv20"])
            & (g.coverage_252 >= pcfg["min_coverage"])
            & (g.coverage_63 >= pcfg["min_coverage"])
            & (g.coverage_21 >= pcfg["min_coverage"])
        ]
        g = g.loc[np.isfinite(g[FEATURES].to_numpy(dtype=float)).all(axis=1)]
        g = g.sort_values(["adv_20", "ticker"], ascending=[False, True]).head(
            pcfg["liquid_universe_size"]
        )
        audit.append({"decision_date": date, "eligible": len(g), "ok": len(g) >= pcfg["min_eligible_stocks"]})
        if len(g) < pcfg["min_eligible_stocks"]:
            continue
        records.append(g)

    if not records:
        raise RuntimeError("No eligible extension candidate dates")
    top300 = pd.concat(records, ignore_index=True)

    # Labels: next-open to next-rebalance-open, computed by the preserved module.
    labeled = attach_labels(top300, panel, pcfg)
    keep = [
        "date",
        "decision_date",
        "execution_date",
        "label_end_date",
        "ticker",
        "split",
        "target_forward_return",
        "spy_return",
        "qqq_return",
        "entry_adj_open",
        "exit_adj_open",
        "close",
        "adv_20",
        "coverage_21",
        "coverage_63",
        "coverage_252",
    ] + FEATURES
    labeled = labeled[keep].copy()
    labeled = _parse_dates(labeled)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_parquet(out_path, index=False)
    audit_df = pd.DataFrame(audit)
    audit_path = out_path.with_name("extension_screening_audit.csv")
    audit_df.to_csv(audit_path, index=False)
    write_json(
        out_path.with_name("extension_panel_meta.json"),
        {
            "source_full_feature_panel": panel_path.relative_to(source_root).as_posix(),
            "source_sha256": sha256_file(panel_path),
            "n_rows": int(len(labeled)),
            "n_decision_dates": int(labeled.decision_date.nunique()),
            "decision_start": str(labeled.decision_date.min().date()),
            "decision_end": str(labeled.decision_date.max().date()),
            "rule": "ADV20 top 300 among price/liquidity/coverage/feature-eligible names, labels via next-open to next-rebalance-open",
        },
    )
    return labeled


def rebuild_model_ready(cfg: Mapping[str, Any], frequency: Optional[str] = None) -> pd.DataFrame:
    """Rebuild the model-ready panel from ``data/raw/market_data.parquet``.

    This calls the preserved authoritative pipeline end to end (features,
    3x3 screening, next-open labels).  It is only needed when the packaged
    derived panel must be regenerated from raw vendor data.
    """
    from .pipeline import run as run_preserved_pipeline

    raw_path = resolve(cfg["data"]["raw_market_data"])
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw market data not found: {raw_path}. Copy the vendor snapshot "
            "from the original project before rebuilding."
        )
    raw = pd.read_parquet(raw_path)
    pcfg = pipeline_config(cfg, frequency)
    panel, screened, ready, audit = run_preserved_pipeline(pcfg, raw)
    print(
        f"Rebuilt {pcfg['rebalance_frequency']}: raw={len(raw):,} panel={len(panel):,} "
        f"screened={len(screened):,} ready={len(ready):,}"
    )
    return ready


def load_extension_panel(cfg: Mapping[str, Any]) -> pd.DataFrame:
    path = resolve(cfg["data"]["extension_panel"])
    if not path.exists():
        raise FileNotFoundError(
            f"Extension candidate panel not found: {path}. "
            "Build it with: python -m src.data_pipeline --build-extension --source-root <original_project>"
        )
    df = pd.read_parquet(path)
    df = _parse_dates(df)
    if df.duplicated(["decision_date", "ticker"]).any():
        raise ValueError("Duplicate decision_date/ticker in extension panel")
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Data preparation and validation")
    parser.add_argument("--build-extension", action="store_true")
    parser.add_argument("--rebuild-panel", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--source-root", type=str, default=None)
    parser.add_argument("--frequency", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    ensure_output_dirs(cfg)
    status = 0
    if args.rebuild_panel:
        rebuild_model_ready(cfg, args.frequency)
        args.build_extension = True
        args.source_root = str(ROOT)
    if args.build_extension:
        if not args.source_root:
            raise SystemExit("--source-root is required with --build-extension")
        panel = build_extension_panel(cfg, Path(args.source_root), args.frequency, args.force)
        print(f"Extension panel: {len(panel):,} rows, {panel.decision_date.nunique()} dates")
    if args.validate:
        df = load_model_ready(cfg, args.frequency)
        splits, summary, excluded = chronological_splits(df, cfg)
        report = data_quality_report(df, splits)
        write_json(resolve(cfg["paths"]["validation"]) / "data_quality.json", report)
        summary.to_csv(resolve(cfg["paths"]["validation"]) / "split_summary.csv", index=False)
        excluded.to_csv(resolve(cfg["paths"]["validation"]) / "label_embargo_excluded.csv", index=False)
        print(json.dumps(report, indent=2))
        print(summary.to_string(index=False))
        status = 0 if report["pass"] else 1
    if not (args.build_extension or args.validate):
        parser.print_help()
    return status


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
