"""Final refit on 2010-2021 with hyperparameters frozen at selection time.

Only labels that are fully realized before the 2022-01-01 test boundary are
used.  The selected model's hyperparameters are taken from the frozen
selection contract; the other three models are refitted with their own
validation-selected configurations so that the pre-planned descriptive OOS
comparison remains fair.  No hyperparameter is tuned here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import joblib
import numpy as np
import pandas as pd

from .data_pipeline import (
    chronological_splits,
    ensure_output_dirs,
    load_config,
    load_model_ready,
    resolve,
)
from .research_policy import active_contract
from .models import MODEL_LABELS, MODEL_NAMES, build_model, params_to_json
from .pipeline import FEATURES
from .utils import environment_info, sha256_file, utc_now, write_json

TARGET = "target_forward_return"


def load_validation_best_params(cfg: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    path = resolve(cfg["paths"]["validation"]) / "validation_best_params.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing validation best-parameter record: {path}")
    payload = json.loads(path.read_text())
    return {
        name: {
            "params": payload["models"][name]["params"],
            "config_index": payload["models"][name]["config_index"],
            "validation_mse": payload["models"][name]["validation_mse"],
        }
        for name in MODEL_NAMES
    }


def refit_models(
    cfg: Optional[Mapping[str, Any]] = None,
    frequency: Optional[str] = None,
    contract: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = load_config() if cfg is None else cfg
    frequency = frequency or cfg["project"]["frequency"]
    ensure_output_dirs(cfg)
    contract = dict(contract) if contract is not None else active_contract(cfg)

    df = load_model_ready(cfg, frequency)
    splits, split_summary, _ = chronological_splits(df, cfg)
    refit = splits["refit"]
    best = load_validation_best_params(cfg)

    selected = contract["selected_model"]
    if selected not in MODEL_NAMES:
        raise RuntimeError(f"Selected model {selected!r} is not one of {MODEL_NAMES}")
    if params_to_json(contract["selected_hyperparameters"]) != params_to_json(best[selected]["params"]):
        raise RuntimeError("Frozen selection hyperparameters disagree with the validation search record")

    # Leakage guards: nothing at or after the test boundary may enter the refit.
    if refit.decision_date.max() >= pd.Timestamp(cfg["data"]["test_start"]):
        raise AssertionError("Refit contains test-period decision dates")
    if refit.label_end_date.max() > pd.Timestamp(cfg["data"]["refit_end_exclusive"]):
        raise AssertionError("Refit contains labels that settle after the refit cutoff")

    seed = int(cfg["project"]["random_seed"])
    model_dir = resolve(cfg["paths"]["models"]) / "refit"
    model_dir.mkdir(parents=True, exist_ok=True)

    estimators: Dict[str, Any] = {}
    summary_rows = []
    for name in MODEL_NAMES:
        params = best[name]["params"]
        estimator = build_model(name, params, seed)
        estimator.fit(refit[FEATURES], refit[TARGET])
        path = model_dir / f"{name}.joblib"
        joblib.dump(estimator, path)
        estimators[name] = estimator
        summary_rows.append(
            {
                "model": name,
                "model_label": MODEL_LABELS[name],
                "is_selected": name == selected,
                "hyperparameters": params_to_json(params),
                "n_training_observations": int(len(refit)),
                "n_decision_dates": int(refit.decision_date.nunique()),
                "n_features": len(FEATURES),
                "refit_decision_start": str(refit.decision_date.min().date()),
                "refit_decision_end": str(refit.decision_date.max().date()),
                "refit_label_end_max": str(refit.label_end_date.max().date()),
                "train_rows": int(len(splits["train"])),
                "validation_rows": int(len(splits["validation"])),
                "validation_mse": best[name]["validation_mse"],
                "random_seed": seed,
                "artifact": str(path.relative_to(resolve(cfg["paths"]["results"]))),
                "artifact_sha256": sha256_file(path),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(resolve(cfg["paths"]["validation"]) / "final_refit_summary.csv", index=False)

    in_sample = {}
    for name in MODEL_NAMES:
        pred = estimators[name].predict(refit[FEATURES])
        in_sample[name] = {
            "mse": float(np.mean((refit[TARGET].to_numpy() - pred) ** 2)),
            "mae": float(np.mean(np.abs(refit[TARGET].to_numpy() - pred))),
        }

    metadata = {
        "stage": "final_refit",
        "generated_utc": utc_now(),
        "frequency": frequency,
        "target": TARGET,
        "feature_names": FEATURES,
        "n_features": len(FEATURES),
        "random_seed": seed,
        "refit_period": {
            "decision_start": str(refit.decision_date.min().date()),
            "decision_end": str(refit.decision_date.max().date()),
            "label_end_max": str(refit.label_end_date.max().date()),
            "cutoff_exclusive": cfg["data"]["refit_end_exclusive"],
            "label_embargo": cfg["data"]["label_embargo"],
        },
        "n_training_observations": int(len(refit)),
        "n_decision_dates": int(refit.decision_date.nunique()),
        "selected_model": selected,
        "selection_criterion": contract["selection_criterion"],
        "selection_run_id": contract.get("run_id"),
        "selection_timestamp_utc": contract.get("selection_timestamp_utc"),
        "test_data_accessed": False,
        "models": {row["model"]: row for row in summary_rows},
        "in_sample_diagnostics": in_sample,
        "package_versions": environment_info()["packages"],
    }
    write_json(resolve(cfg["paths"]["validation"]) / "final_model_metadata.json", metadata)

    return {
        "estimators": estimators,
        "summary": summary,
        "metadata": metadata,
        "refit_rows": int(len(refit)),
        "selected_model": selected,
    }


def load_refit_models(cfg: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    cfg = load_config() if cfg is None else cfg
    model_dir = resolve(cfg["paths"]["models"]) / "refit"
    models = {}
    for name in MODEL_NAMES:
        path = model_dir / f"{name}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Missing refitted model {path}. Run the refit stage first.")
        models[name] = joblib.load(path)
    return models


if __name__ == "__main__":  # pragma: no cover
    result = refit_models()
    print(result["summary"][["model", "is_selected", "hyperparameters", "n_training_observations"]].to_string(index=False))
