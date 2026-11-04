"""Exploratory robustness diagnostics for Appendix D of the report.

These variants are computed AFTER the single confirmatory test evaluation in
run_backtest.py, using the frozen main configuration (lambda from validation,
gamma from the fitting sample, pre-set position rule). They are NOT used to
re-select lambda, features, the cap, or the cost assumption.

Variants:
  cost 0.5 / 1 / 2 bp per unit of target-weight turnover;
  position cap +/-1.0 instead of +/-1.5;
  long-only positions clipped to [0, 1.5].

Output: output/results/robustness.csv
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from features import build_dataset
from run_backtest import (
    PI_MAX, TRAIN_END, VAL_END, calibrate_gamma, perf, positions, split_dataset,
    standardize,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "output", "results")


def net_sharpe(pi: np.ndarray, exs: np.ndarray, rf: np.ndarray, cost: float) -> dict:
    turnover = np.abs(np.diff(np.concatenate([[0.0], pi])))
    net = rf + pi * exs - cost * turnover
    p = perf(net, rf)
    return {"ann_return": p["ann_return"], "ann_vol": p["ann_vol"], "sharpe": p["sharpe"]}


def main() -> None:
    with open(os.path.join(RESULTS, "run_config.json")) as f:
        cfg = json.load(f)
    lam = cfg["lambda_selected"]

    ds = build_dataset().dropna(subset=["naive_excess"]).reset_index(drop=True)
    _train, _val, test, trainval = split_dataset(ds)
    Xtv_s, Xte_s = standardize(trainval, trainval, test)
    ridge = Ridge(alpha=lam).fit(Xtv_s, trainval["fwd_excess"])
    gamma = calibrate_gamma(ridge.predict(Xtv_s), trainval["sigma2_21"].to_numpy())
    yhat = ridge.predict(Xte_s)

    sig2 = test["sigma2_21"].to_numpy()
    exs = test["fwd_excess"].to_numpy()
    rf = test["rf_next"].to_numpy()
    raw = yhat / (gamma * sig2)

    rows = []
    for cost_bp, cost in [(0.5, 0.5e-4), (1.0, 1e-4), (2.0, 2e-4)]:
        rows.append({"variant": f"cost {cost_bp} bp",
                     **net_sharpe(np.clip(raw, -PI_MAX, PI_MAX), exs, rf, cost)})
    for cap in [1.0, 1.5]:
        rows.append({"variant": f"position cap +/-{cap}",
                     **net_sharpe(np.clip(raw, -cap, cap), exs, rf, 1e-4)})
    rows.append({"variant": "long-only [0, 1.5]",
                 **net_sharpe(np.clip(raw, 0.0, PI_MAX), exs, rf, 1e-4)})

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "robustness.csv"), index=False)
    print("[exploratory diagnostics -- not used for model selection]")
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
