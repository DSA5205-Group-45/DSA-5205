# Classmate GitHub Alignment Changelog

## Scope

This revision follows `Project2_主实验与Extension代码修改指示.docx`. The classmate GitHub version is the numerical and strategy-semantics baseline. Engineering improvements such as relative paths, `nbclient`, automated execution, and additional QA are retained.

The pre-revision code, notebooks, report material, and outputs are backed up in `../DSA5205_Project2_before_classmate_alignment/`. The directory was not a Git repository, so no Git branch was created. `baseline_manifest.json` records the original source and data hashes.

## Changes

### Main experiment: fixed default strategy, no numerical change

- Kept the original 30-stock double-sorted candidate pool and the same 15 features and `target_forward_return` label.
- Kept Ridge, LASSO, Random Forest, and XGBoost with their existing small grids, validation-MSE selection, seed 42, and realized-label cutoffs.
- Kept one default portfolio rule for all models: prediction descending, ticker ascending, Top 10, 10% per stock, next-open execution, and 20 bps per traded side.
- Kept `momentum_top10`, `equal_weight_30`, SPY, and QQQ as common benchmarks, not as candidates for ex-post replacement of the default strategy.
- Verified byte-for-byte equality of the main comparison CSV files before and after this revision. No main-experiment numerical result changed.

### Extension: restored the classmate 60-stock pool

Corrected `strategies.py::pools()` after sorting the eligible Top 300 by momentum and splitting it into three approximately equal groups:

- High: highest 20 by `mom_2_12`, with ticker ascending for ties.
- Medium: closest 20 to the group median `mom_2_12`, ordered by distance and then ticker.
- Low: lowest 20 by `mom_2_12`, with ticker ascending for ties.

The previous implementation incorrectly used `head(20)` for all three groups, which selected the highest-momentum 20 within Medium and Low and changed the Ridge training sample. The s30 momentum–volatility double sort remains unchanged and matches the main candidate pool on every real decision date.

### Extension: preserved independent nine-strategy research

- Retained B1, B2, B3, R1, R2, R3, R4, SPY, and QQQ under the classmate selection and backtest semantics.
- R1 uses the same stocks as B1, nonnegative clipped prediction weights, iterative 20% caps, and equal-weight fallback when all scores are nonpositive.
- R4 uses the s60-trained extension Ridge to score s30; it is not renamed or substituted for main-experiment `model_top10`.
- Main outputs remain under `outputs/four_model_comparison/monthly` and `outputs/models/monthly`; extension outputs remain under `outputs/ridge_nine_strategies/monthly`.
- Old generated outputs were backed up and then regenerated from the same fixed raw-data snapshot.

### Tests and documentation

- Added `tests/test_strategies.py` for exact High/Medium/Low membership, pool sizes, R1/B1 membership, weight caps, nonpositive-score fallback, and model-independent momentum baselines.
- Added `tests/test_default_strategy.py` for the fixed Top-10 equal-weight rule, ticker tie-breaking, equal-weight candidates, and one-time ETF entry costs.
- Updated `README.md` with the exact s60 selection rules, output separation, and test command.
- Updated `report_sections_4.4_and_interpretation.md` so all extension numbers and interpretation come from the corrected run. R1 and R2 are not presented as main-default-strategy successes.
- Added `requirements-lock.txt` and `REPRODUCIBILITY_RECORD.md`.

## Alignment result

The corrected run reproduces the classmate reference values: Ridge alpha 1000; train/validation/refit rows 7080/1320/8520; test MSE approximately 0.0170500242; MAE approximately 0.0918752271; R² approximately −0.0067992492; mean Rank IC approximately 0.0218000370; and all nine strategies' 2022–2025 annual returns to the displayed two-decimal percentages. Remaining differences are limited to floating-point precision and display rounding, as documented in the reproduction record.
