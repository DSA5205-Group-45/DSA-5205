"""Shared fixtures.

Most tests run on a small synthetic panel so they are fast, deterministic and
independent of whether the proprietary price snapshot is present.  A few
integration tests use the real model-ready panel and skip cleanly if it is
absent.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline import load_config, load_model_ready  # noqa: E402
from src.pipeline import FEATURES  # noqa: E402


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def real_cfg():
    return load_config()


@pytest.fixture(scope="session")
def real_model_ready(real_cfg):
    try:
        return load_model_ready(real_cfg, "monthly")
    except FileNotFoundError:
        pytest.skip("Real model-ready dataset is not available in this checkout")


def _make_panel(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = {
        "train": ["2019-01-31", "2019-02-28", "2019-03-29", "2019-04-30"],
        "validation": ["2020-01-31", "2020-02-28", "2020-03-31", "2020-04-30"],
        "test": ["2022-01-31", "2022-02-28", "2022-03-31", "2022-04-29"],
    }
    tickers = [f"T{i:02d}" for i in range(30)]
    rows = []
    for split, split_dates in dates.items():
        for date in split_dates:
            decision = pd.Timestamp(date)
            execution = decision + pd.Timedelta(days=2)
            label_end = decision + pd.Timedelta(days=32)
            spy_return = float(rng.normal(0.005, 0.02))
            qqq_return = float(rng.normal(0.006, 0.025))
            for i, ticker in enumerate(tickers):
                features = rng.normal(0.0, 0.5, len(FEATURES))
                target = float(rng.normal(0.01, 0.06))
                row = {
                    "decision_date": decision,
                    "execution_date": execution,
                    "label_end_date": label_end,
                    "date": decision,
                    "ticker": ticker,
                    "split": split,
                    "target_forward_return": target,
                    "target_absolute_return": target,
                    "spy_return": spy_return,
                    "qqq_return": qqq_return,
                    "momentum_group": ["High", "Medium", "Low"][i % 3],
                    "volatility_group": ["Low", "Medium", "High"][i % 3],
                    "close": 50.0 + i,
                    "adv_20": 1e8,
                }
                row.update({feature: float(value) for feature, value in zip(FEATURES, features)})
                rows.append(row)
    frame = pd.DataFrame(rows)
    frame["target_spy_relative_return"] = frame["target_forward_return"] - frame["spy_return"]
    frame["target_qqq_relative_return"] = frame["target_forward_return"] - frame["qqq_return"]
    return frame.sort_values(["decision_date", "ticker"]).reset_index(drop=True)


@pytest.fixture(scope="session")
def synthetic_panel() -> pd.DataFrame:
    return _make_panel()


@pytest.fixture
def synthetic_cfg(tmp_path, real_cfg, synthetic_panel):
    """Config whose data and output paths live entirely under ``tmp_path``."""
    cfg = copy.deepcopy(real_cfg)
    data_dir = tmp_path / "data" / "processed_ready" / "monthly"
    data_dir.mkdir(parents=True, exist_ok=True)
    synthetic_panel.to_parquet(data_dir / "model_ready.parquet", index=False)
    cfg["data"]["model_ready_dir"] = str(tmp_path / "data" / "processed_ready")
    base = tmp_path / "results"
    cfg["paths"] = {
        "results": str(base),
        "validation": str(base / "validation"),
        "validation_figures": str(base / "validation" / "figures"),
        "oos": str(base / "oos"),
        "oos_figures": str(base / "oos" / "figures"),
        "ridge_extension": str(base / "extensions"),
        "ridge_extension_figures": str(base / "extensions" / "figures"),
        "models": str(base / "models"),
        "research_diagnostics": str(base / "diagnostics"),
        "eight_strategy": str(base / "oos" / "eight_strategy"),
    }
    cfg["_tmp_path"] = str(tmp_path)
    return cfg


@pytest.fixture(scope="session")
def synthetic_predictions() -> pd.DataFrame:
    """30 candidates over four periods, shaped like a backtest input."""
    rng = np.random.default_rng(11)
    rows = []
    dates = pd.to_datetime(["2020-01-31", "2020-02-28", "2020-03-31", "2020-04-30"])
    for date in dates:
        for i in range(30):
            rows.append(
                {
                    "decision_date": date,
                    "execution_date": date + pd.Timedelta(days=2),
                    "label_end_date": date + pd.Timedelta(days=32),
                    "ticker": f"T{i:02d}",
                    "prediction": float(rng.normal(0.0, 0.05)),
                    "mom_2_12": float(rng.normal(0.0, 0.2)),
                    "target_forward_return": float(rng.normal(0.01, 0.06)),
                    "spy_return": 0.005,
                    "qqq_return": 0.006,
                    "split": "validation",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def selection_fixture():
    """Minimal validation comparison table used by selection tests."""
    return pd.DataFrame(
        {
            "model": ["ridge", "lasso", "rf", "xgb"],
            "best_params": [
                '{"alpha": 1000.0}',
                '{"alpha": 0.001}',
                '{"max_depth": 3, "min_samples_leaf": 25}',
                '{"learning_rate": 0.03, "max_depth": 2, "n_estimators": 100}',
            ],
            "grid_candidates": [6, 5, 4, 8],
            "validation_mse": [0.0224, 0.0222, 0.0225, 0.0227],
            "validation_rmse": [0.1497, 0.1490, 0.1500, 0.1507],
            "validation_mae": [0.0911, 0.0910, 0.0912, 0.0917],
            "validation_r2_sklearn": [0.039, 0.045, 0.038, 0.029],
            "validation_r2_oos_trainmean": [0.043, 0.049, 0.042, 0.034],
            "validation_sharpe": [0.92, 0.93, 0.86, 0.69],
            "validation_cagr": [0.45, 0.46, 0.40, 0.27],
            "validation_max_drawdown": [-0.32, -0.30, -0.33, -0.29],
            "validation_mean_rank_ic": [0.04, 0.02, 0.03, -0.01],
        }
    )


@pytest.fixture
def selection_cfg(tmp_path, real_cfg, selection_fixture):
    cfg = copy.deepcopy(real_cfg)
    base = tmp_path / "results"
    validation = base / "validation"
    validation.mkdir(parents=True, exist_ok=True)
    selection_fixture.to_csv(validation / "validation_model_comparison.csv", index=False)
    cfg["paths"] = {
        "results": str(base),
        "validation": str(validation),
        "validation_figures": str(validation / "figures"),
        "oos": str(base / "oos"),
        "oos_figures": str(base / "oos" / "figures"),
        "ridge_extension": str(base / "extensions"),
        "ridge_extension_figures": str(base / "extensions" / "figures"),
        "models": str(base / "models"),
        "research_diagnostics": str(base / "diagnostics"),
        "eight_strategy": str(base / "oos" / "eight_strategy"),
    }
    return cfg
