"""Four-model interface: Ridge, LASSO, Random Forest, XGBoost.

The factories and the default hyperparameter grids are inherited from the
authoritative pipeline (``src/models.py`` of DSA5205_Project2_Pipeline).  The
only grid change is that XGBoost ``n_estimators`` is treated as a tuned
dimension ``{100, 200}`` instead of being fixed at 200, so that the grid
contains ``max_depth``, ``learning_rate`` *and* ``n_estimators`` as required by
the project brief.  Ridge, LASSO and Random Forest grids are untouched.

All models are fitted through a single ``fit_model`` entry point so that the
four algorithms see exactly the same rows, features and target.
"""
from __future__ import annotations

import itertools
import json
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .pipeline import FEATURES

MODEL_NAMES: List[str] = ["ridge", "lasso", "rf", "xgb"]
MODEL_LABELS: Dict[str, str] = {
    "ridge": "Ridge Regression",
    "lasso": "LASSO Regression",
    "rf": "Random Forest",
    "xgb": "XGBoost",
}
PRIMARY_MODEL = "ridge"

# Default grids.  Kept as a module constant for backward compatibility with
# the original ``run_model`` helper; the pipeline itself expands the grids
# from ``config/experiment_config.yaml`` so that the experiment is fully
# documented in one place.
GRIDS: Dict[str, List[Dict[str, Any]]] = {
    "ridge": [{"alpha": a} for a in [0.1, 1, 10, 100, 1000, 10000]],
    "lasso": [{"alpha": a} for a in [0.00001, 0.0001, 0.001, 0.01, 0.1]],
    "rf": [
        {"max_depth": d, "min_samples_leaf": leaf}
        for d in [3, 6]
        for leaf in [10, 25]
    ],
    "xgb": [
        {"max_depth": d, "learning_rate": lr, "n_estimators": n}
        for d in [2, 3]
        for lr in [0.03, 0.1]
        for n in [100, 200]
    ],
}


def parameter_grid(name: str, model_config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Expand a model's declared grid deterministically.

    The order of the keys follows the YAML declaration; the last key varies
    fastest.  The resulting list is stable across runs and platforms.
    """
    grid = dict(model_config.get("grid", {}))
    if not grid:
        return [{}]
    keys = list(grid.keys())
    values = [list(grid[k]) for k in keys]
    expanded: List[Dict[str, Any]] = []
    for combo in itertools.product(*values):
        expanded.append({k: _native(v) for k, v in zip(keys, combo)})
    return expanded


def _native(value: Any) -> Any:
    """Convert numpy scalars to plain Python so JSON serialisation is stable."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def grid_size(name: str, config: Mapping[str, Any]) -> int:
    return len(parameter_grid(name, config["models"][name]))


def total_grid_size(config: Mapping[str, Any]) -> int:
    return sum(grid_size(n, config) for n in MODEL_NAMES)


def build_model(name: str, params: Mapping[str, Any], seed: int = 42):
    """Construct one unfitted estimator with the original preprocessing rules.

    Ridge and LASSO use ``StandardScaler`` inside a scikit-learn ``Pipeline``
    so that scaling parameters are estimated only on the data passed to
    ``fit``.  Tree models do not need scaling and are left raw.
    """
    params = {k: _native(v) for k, v in dict(params).items()}
    if name == "ridge":
        return Pipeline([("scale", StandardScaler()), ("model", Ridge(**params))])
    if name == "lasso":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", Lasso(**params, max_iter=20000, random_state=seed)),
            ]
        )
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=250,
            max_features=0.7,
            n_jobs=-1,
            random_state=seed,
            **params,
        )
    if name == "xgb":
        from xgboost import XGBRegressor

        return XGBRegressor(
            objective="reg:squarederror",
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=10,
            tree_method="hist",
            random_state=seed,
            n_jobs=4,
            **params,
        )
    raise ValueError(f"Unknown model: {name!r}")


def fit_model(name: str, params: Mapping[str, Any], X, y, seed: int = 42):
    model = build_model(name, params, seed)
    model.fit(X, y)
    return model


def metrics(y: Iterable[float], pred: Iterable[float]) -> Dict[str, float]:
    """Original in-sample / out-of-sample prediction metrics."""
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return {
        "mse": float(mean_squared_error(y, pred)),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
    }


def convergence_info(name: str, estimator) -> Dict[str, Any]:
    """Record solver convergence and sparsity for the linear models."""
    info: Dict[str, Any] = {
        "converged": np.nan,
        "n_iter": np.nan,
        "n_nonzero_features": np.nan,
    }
    if name in ("ridge", "lasso"):
        inner = estimator.named_steps["model"] if hasattr(estimator, "named_steps") else estimator
        coef = np.asarray(inner.coef_, dtype=float).ravel()
        info["n_nonzero_features"] = int(np.sum(np.abs(coef) > 1e-12))
        if name == "lasso":
            n_iter = getattr(inner, "n_iter_", None)
            if isinstance(n_iter, (int, np.integer)):
                info["n_iter"] = int(n_iter)
                info["converged"] = bool(int(n_iter) < int(inner.max_iter))
    return info


def params_to_json(params: Mapping[str, Any]) -> str:
    return json.dumps({k: _native(v) for k, v in params.items()}, sort_keys=True)


def select_best_config(
    search: pd.DataFrame, criterion: str = "lowest_validation_mse"
) -> pd.Series:
    """Pick the best configuration from a hyperparameter search table.

    Ties are broken by candidate index (the deterministic grid order), never
    by any test-period information.
    """
    if search.empty:
        raise ValueError("Empty hyperparameter search table")
    if criterion == "lowest_validation_mse":
        key = ["validation_mse", "config_index"]
        ascending = [True, True]
    elif criterion == "highest_validation_sharpe":
        key = ["validation_sharpe", "config_index"]
        ascending = [False, True]
    else:
        raise ValueError(f"Unknown selection criterion: {criterion!r}")
    return search.sort_values(key, ascending=ascending).iloc[0]


# ---------------------------------------------------------------------------
# Legacy single-model entry point (kept so that the original ``run_models.py``
# continues to work).  New code should use ``src.validation`` instead.
# ---------------------------------------------------------------------------
def run_model(frequency: str = "monthly", model: str = "ridge", seed: int = 42):
    from .pipeline import ROOT

    data_path = ROOT / "data/processed_ready" / frequency / "model_ready.parquet"
    df = pd.read_parquet(data_path).copy()
    if df.empty:
        raise RuntimeError("Empty model-ready dataset")
    df["decision_date"] = pd.to_datetime(df.decision_date)
    df["label_end_date"] = pd.to_datetime(df.label_end_date)
    if any(df.ticker.astype(str).str.startswith("SYN")):
        raise RuntimeError("Synthetic input refused")
    target = "target_forward_return"
    features = FEATURES
    tr = df[(df.split == "train") & (df.label_end_date <= pd.Timestamp("2020-01-01"))].copy()
    va = df[(df.split == "validation") & (df.label_end_date <= pd.Timestamp("2022-01-01"))].copy()
    te = df[df.split == "test"].copy()
    incomplete = te.groupby("decision_date").ticker.nunique()
    if (incomplete != 30).any():
        raise RuntimeError("Test date has fewer than 30 labeled candidates.")
    if min(len(tr), len(va), len(te)) == 0:
        raise RuntimeError("Empty chronological split")
    history = []
    for params in GRIDS[model]:
        estimator = build_model(model, params, seed)
        estimator.fit(tr[features], tr[target])
        a = metrics(tr[target], estimator.predict(tr[features]))
        b = metrics(va[target], estimator.predict(va[features]))
        history.append(
            {
                "model": model,
                "params": json.dumps(params, sort_keys=True),
                **{"train_" + k: v for k, v in a.items()},
                **{"validation_" + k: v for k, v in b.items()},
            }
        )
    tuning = pd.DataFrame(history).sort_values(["validation_mse", "params"])
    best = json.loads(tuning.iloc[0]["params"])
    final = build_model(model, best, seed)
    refit = df[(df.decision_date < pd.Timestamp("2022-01-01")) & (df.label_end_date <= pd.Timestamp("2022-01-01"))]
    final.fit(refit[features], refit[target])
    te["prediction"] = final.predict(te[features])
    te["model"] = model
    te["pred_rank"] = te.groupby("decision_date")["prediction"].rank(method="first", ascending=False)
    test_metrics = metrics(te[target], te.prediction)
    out = ROOT / "outputs" / "models" / frequency / model
    out.mkdir(parents=True, exist_ok=True)
    tuning.to_csv(out / "training_validation_grid.csv", index=False)
    te.to_parquet(out / "test_predictions.parquet", index=False)
    te[
        [
            "decision_date",
            "ticker",
            "model",
            "prediction",
            "pred_rank",
            "target_forward_return",
            "execution_date",
            "label_end_date",
        ]
    ].to_csv(out / "test_predictions.csv", index=False)
    (out / "model_summary.json").write_text(
        json.dumps(
            {
                "best_params": best,
                "test_metrics": test_metrics,
                "train_rows": len(tr),
                "validation_rows": len(va),
                "refit_rows": len(refit),
                "test_rows": len(te),
                "seed": seed,
                "primary_target": target,
            },
            indent=2,
        )
    )
    return te, tuning, final
