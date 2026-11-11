# Requirements Compliance Checklist

Mapping of the DSA5205 Project 2 instructions (Sections 4.1–4.9, interpretation, reproducibility and deliverables) and the marking rubric to the report, notebook and supporting outputs.

| Requirement | Report section | Notebook section | Supporting output | Status |
|---|---|---|---|---|
| **§4.1 Markets and assets** — market, assets, frequency, calendar, currency | §2.1–2.2 | §2 | `results/oos/eight_strategy/metrics.csv` | Complete |
| **§4.1 Trading times** — decision vs execution | §2.1, §9.1 | §2, §8 | `validation_predictions.csv` (`decision_date`, `execution_date`, `label_end_date`) | Complete |
| **§4.1 Investable benchmark** | §2.4, §4.1 | §2, §10 | SPY/QQQ monthly returns | Complete |
| **§4.1 Lawful data source and corporate actions** | §2.3 | §3, §4 | preserved `src/pipeline.py`; `data_quality.json` | Complete |
| **§4.2 Label horizon and alignment** | §3.1 | §4 | `label_embargo_excluded.csv`; `validation_predictions.csv` | Complete |
| **§4.2 Features, formulas, lookbacks** | §3.2, Table 2 | §4 | `src/features.py`; feature table | Complete |
| **§4.2 Features known at decision time** | §3.1, §3.5 | §4 | `decision_date < execution_date < label_end_date` test | Complete |
| **§4.2 Scaling fitted on past data only** | §3.5 | §4 | `test_scaler_is_fitted_on_training_rows_only` | Complete |
| **§4.2 Cleaning and missing-data policy** | §3.4 | §4 | `data_quality.json` | Complete |
| **§4.3 Buy-and-hold benchmarks** | §4.1 | §2, §10 | SPY, QQQ series | Complete |
| **§4.3 Forecast / strategy baselines** | §4.2 | §2, §10 | B1, B3 series | Complete |
| **§4.4 Economic story for the model** | §5.1 | §1, §5 | `primary_model_contract.json` rationale | Complete |
| **§4.4 Overfitting control** | §5.5, §7.3 | §6 | 23-configuration grid; seeds | Complete |
| **§4.4 Model comparison** | §5.4, §8.2 | §5–7 | `validation_model_comparison.csv` | Complete |
| **§4.4 Correlated features and regularisation** | §5.2, Table 3 | §5 | correlation matrix and strong pairs | Complete |
| **§4.5 Training / validation / testing split** | §6, Table 5 | §4, §6 | `split_summary.csv` | Complete |
| **§4.5 Leakage prevention and embargo** | §6 | §4, §7 | `label_embargo_excluded.csv`; leakage tests | Complete |
| **§4.5 Refit methodology** | §6, §8.5 | §8 | `final_refit_summary.csv`; `final_model_metadata.json` | Complete |
| **§4.6 Training methods** | §7.1 | §6 | `src/models.py`, `src/validation.py` | Complete |
| **§4.6 Packages, seeds, exact settings** | §7.2, §15 | §14 | `requirements.txt`; run metadata JSON | Complete |
| **§4.7 Time-respecting validation** | §8.1 | §6–7 | `validation_run_metadata.json` | Complete |
| **§4.7 Small pre-declared grid** | §7.3, Table 4 | §6 | `validation_hyperparameter_search.csv` | Complete |
| **§4.7 One clear selection rule** | §8.4–8.5 | §7 | `primary_model_contract.json` | Complete |
| **§4.7 Historical selection audit preserved** | §8.3 | §7 | `results/validation/archive/*` | Complete |
| **§4.8 Score-to-position rule** | §9.1 | §8 | `results/oos/oos_predictions.csv` | Complete |
| **§4.8 Execution and costs** | §9.2 | §8 | `results/oos/eight_strategy/monthly_returns.csv` | Complete |
| **§4.8 Constraints and risk limits** | §9.3 | §8 | no leverage, 10% weights | Complete |
| **§4.9 Correspondence between training and testing** | §10.1 | §8–9 | alignment checks JSON | Complete |
| **§4.9 Annualised return and Sharpe** | §10.3, Table 9 | §10 | `eight_strategy/metrics.csv` | Complete |
| **§4.9 Comparison to benchmarks** | §10.3–10.6 | §10 | Figures 2–7 | Complete |
| **§4.9 Out-of-sample plots** | Figures 2–7 | §9–10 | `results/oos/figures/fig1…fig6` | Complete |
| **Interpretation: what worked / did not** | §11.1–11.2 | §11 | report text | Complete |
| **Interpretation: limitations** | §13 | §11 | report text | Complete |
| **Reproducibility: exact commands** | §15.1 | §14 | `README.md`, `run_pipeline.py`, `run_tests.py` | Complete |
| **Reproducibility: directory structure** | §15.1 | §14 | `README.md` | Complete |
| **Reproducibility: fixed seeds and relative paths** | §7.2, §15 | §14 | config, tests | Complete |
| **Deliverable: README with run commands** | — | — | `README.md` | Complete |
| **Deliverable: requirements file** | — | — | `requirements.txt` | Complete |
| **Deliverable: runnable notebook** | — | whole notebook | `notebooks/Project2_Final.ipynb` | Complete |
| **Deliverable: code package** | — | — | `src/`, `run_pipeline.py`, `tests/` | Complete |
| **Marking rubric: markets and timing** | §2 | §2 | — | Complete |
| **Marking rubric: labels and features** | §3 | §4 | — | Complete |
| **Marking rubric: benchmarks** | §4 | §10 | — | Complete |
| **Marking rubric: model and overfitting** | §5 | §5 | — | Complete |
| **Marking rubric: data split** | §6 | §7 | — | Complete |
| **Marking rubric: training** | §7 | §6 | — | Complete |
| **Marking rubric: validation and selection** | §8 | §7 | — | Complete |
| **Marking rubric: trades and costs** | §9 | §8 | — | Complete |
| **Marking rubric: out-of-sample evaluation** | §10 | §9–10 | Figures 2–7 | Complete |
| **Marking rubric: interpretation and reproducibility** | §11, §15 | §11, §14 | audit trail | Complete |
| **Required report length (15–25 pages)** | 23 pages | — | `report/Project2_Report.pdf` | Complete |
| **Required figures (score-to-position + six OOS)** | Figures 1–7 | §9–10 | `figures/`; `results/oos/figures/` | Complete |
| **Required model comparison (4 ML models)** | §8.2, §10.2 | §6–7, §9 | `validation_model_comparison.csv` | Complete |
| **Required eight-strategy comparison** | §10.3, Table 9 | §10 | `eight_strategy/metrics.csv` | Complete |
| **Ridge primary, LASSO audit preserved** | §8.3–8.6 | §7 | `primary_model_contract.json`, `archive/` | Complete |
| **Exploratory extension separated** | §12 | §12 | `results/extensions/` | Complete |

| **§4.4 Precise training methods** | §7.1 | §6 | fitted estimator inspection; `INTEGRATION_AUDIT.md` §6 | Complete |
| **§4.7 Ridge assessment without OOS justification** | §8.2–8.6 | §7 | `primary_model_contract.json` | Complete |
| **§4.8 Score-to-position rule** | §9.1, Figure 1 | §8 | `results/oos/score_to_position_example.csv` | Complete |
| **§4.8 Buy and sell costs** | §9.2 | §8 | `test_backtest.py` | Complete |
| **Superseded automatic selection removed** | §8.3 | §7 | `results/validation/archive/superseded_automatic_selection/` | Complete |

## Notes

- Figure 1 is the score-to-position illustration, Figures 2–7 are the six out-of-sample exhibits, and diagnostic figures are Appendix A1–A2.
- The report is 23 pages including references and appendices, within the 15–25 page requirement.
- The superseded automatic LASSO rule is preserved read-only in `results/validation/archive/superseded_automatic_selection/` and is not used in the active workflow. The primary-model contract documents that the Ridge designation is not a clean pre-registration.
- No claim of statistical significance is made for Sharpe differences. The report states that the test window is one historical path.
