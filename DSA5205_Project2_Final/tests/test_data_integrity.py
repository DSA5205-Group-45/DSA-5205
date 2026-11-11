"""Data integrity: schema, dates, uniqueness, finite values and coverage."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_pipeline import (
    REQUIRED_MODEL_COLUMNS,
    chronological_splits,
    data_quality_report,
    load_model_ready,
)
from src.pipeline import FEATURES


def test_required_columns_exist(real_model_ready):
    missing = [c for c in REQUIRED_MODEL_COLUMNS if c not in real_model_ready.columns]
    assert not missing, f"missing required columns: {missing}"
    assert len(FEATURES) == 15


def test_dates_are_parsed_as_datetime(real_model_ready):
    for column in ("decision_date", "execution_date", "label_end_date"):
        assert pd.api.types.is_datetime64_any_dtype(real_model_ready[column])


def test_no_duplicate_stock_date_observations(real_model_ready):
    assert not real_model_ready.duplicated(["decision_date", "ticker"]).any()


def test_no_invalid_or_infinite_feature_values(real_model_ready):
    values = real_model_ready[FEATURES].to_numpy(dtype=float)
    assert np.isfinite(values).all()
    assert not np.isinf(values).any()


def test_sample_counts_and_date_coverage(real_model_ready, real_cfg):
    splits, summary, _ = chronological_splits(real_model_ready, real_cfg)
    counts = real_model_ready.groupby("decision_date").ticker.nunique()
    assert (counts == 30).all()
    assert len(real_model_ready) == 5760
    assert real_model_ready.decision_date.nunique() == 192
    assert len(splits["train"]) == 3540
    assert len(splits["validation"]) == 660
    assert len(splits["test"]) == 1440
    assert splits["train"].decision_date.nunique() == 118
    assert splits["validation"].decision_date.nunique() == 22
    assert splits["test"].decision_date.nunique() == 48
    assert splits["test"].decision_date.min() == pd.Timestamp("2022-01-31")
    assert splits["test"].decision_date.max() == pd.Timestamp("2025-12-31")
    assert set(summary["split"]) == {"train", "validation", "refit", "test"}


def test_label_timing_is_strictly_ordered(real_model_ready):
    assert (real_model_ready.decision_date < real_model_ready.execution_date).all()
    assert (real_model_ready.execution_date < real_model_ready.label_end_date).all()


def test_quality_report_passes_on_real_data(real_model_ready, real_cfg):
    splits, _, _ = chronological_splits(real_model_ready, real_cfg)
    report = data_quality_report(real_model_ready, splits)
    assert report["pass"], report["issues"]
    assert report["n_features"] == 15


def test_quality_report_passes_on_synthetic_data(synthetic_panel, synthetic_cfg):
    splits, _, _ = chronological_splits(synthetic_panel, synthetic_cfg)
    report = data_quality_report(synthetic_panel, splits)
    assert report["pass"], report["issues"]


def test_quality_report_detects_duplicates(synthetic_panel, synthetic_cfg):
    corrupted = pd.concat([synthetic_panel, synthetic_panel.head(1)], ignore_index=True)
    splits, _, _ = chronological_splits(corrupted, synthetic_cfg)
    report = data_quality_report(corrupted, splits)
    assert not report["pass"]
    assert any("duplicate" in issue for issue in report["issues"])


def test_loader_rejects_synthetic_tickers(synthetic_panel, synthetic_cfg):
    frame = synthetic_panel.copy()
    frame.loc[0, "ticker"] = "SYN000"
    path = f"{synthetic_cfg['data']['model_ready_dir']}/monthly/model_ready.parquet"
    frame.to_parquet(path, index=False)
    try:
        load_model_ready(synthetic_cfg, "monthly")
    except ValueError as exc:
        assert "Synthetic" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Synthetic tickers should be rejected")
