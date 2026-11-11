# Reproducibility Record

## Fixed data snapshot

The final run reused the existing real-data snapshot; Yahoo Finance was not refreshed to chase reference results.

- File: `data/raw/market_data.parquet`
- SHA-256: `17beab0e9f1e1d96fe84edce76bd196e0d99bf7fcf53e3caf101ec8b97a47e6f`
- Size: 59,609,198 bytes
- Rows: 2,106,106
- Tickers: 505
- Dates: 2009-01-02 through 2026-10-30
- Download failures: 0
- Universe limitation: current S&P 500 constituents backfilled through time; survivorship and constituent look-ahead bias remain.

The data build produced 5,760 monthly rows over 192 decision dates, 1,920 quarterly rows over 64 dates, and 960 semiannual rows over 32 dates. Every monthly date has 30 candidates. Chronological decision-date splits are train 2010–2019, validation 2020–2021, and test 2022–2025, with realized-label cutoffs enforced during fitting.

## Actual environment

- Python 3.11.15, Windows
- numpy 1.26.4
- pandas 2.3.3
- pyarrow 21.0.0
- scikit-learn 1.9.1
- xgboost 3.2.0
- yfinance 1.8.0
- matplotlib 3.11.2
- nbclient 0.11.0
- nbformat 5.11.1
- pytest 9.2.0
- Random seed: 42

The complete environment is recorded in `requirements-lock.txt`.

## Final commands

From the project root:

```bash
python -m pytest -q
python execute_notebooks.py --notebook all
```

The data notebook ran first, followed by the four-model main experiment and then the Ridge extension. Both notebooks completed with no error outputs. Python syntax compilation also completed successfully.

## Acceptance results

- `pytest`: **5 passed**.
- Raw-data SHA-256 after the final run: unchanged.
- Main validation, prediction, portfolio, reference, and period-return CSV files: byte-for-byte unchanged from the pre-alignment versions.
- All 192 real monthly dates: s60 contains 60 unique tickers with 20 per momentum group; s30 contains 30 unique tickers.
- Extension s30 ticker membership equals the main original 30-stock candidate pool on every real date.
- All four models use identical test decision-date/ticker keys and 48 test periods.
- R1 and B1 hold identical ticker sets on every test date; all strategy weights sum to 1; R1 weights do not exceed 20%.
- Main and extension outputs are stored in separate directories.

## Classmate reference comparison

| Quantity | Final run | Classmate reference |
|---|---:|---:|
| Ridge alpha | 1000 | 1000 |
| Train rows | 7080 | 7080 |
| Validation rows | 1320 | 1320 |
| Refit rows | 8520 | 8520 |
| Test MSE | 0.0170500243656 | approximately 0.0170500242 |
| Test MAE | 0.0918752305155 | approximately 0.0918752271 |
| Test R² | −0.0067992245540 | approximately −0.0067992492 |
| Mean Rank IC | 0.0218000370473 | approximately 0.0218000370 |

Annual net returns match the classmate table to the displayed two-decimal percentages:

| Strategy | 2022 | 2023 | 2024 | 2025 |
|---|---:|---:|---:|---:|
| B1 | 12.33% | 58.61% | 126.14% | 122.99% |
| B2 | 7.04% | 20.77% | 63.29% | 41.24% |
| B3 | 5.63% | 50.35% | 122.68% | 49.00% |
| R1 | 8.42% | 76.25% | 158.85% | 151.93% |
| R2 | −28.00% | 44.07% | 76.18% | 96.55% |
| R3 | −25.35% | 45.45% | 54.62% | 34.25% |
| R4 | −13.03% | 59.36% | 73.88% | 79.48% |
| SPY | −8.82% | 21.42% | 23.86% | 17.72% |
| QQQ | −18.79% | 43.23% | 23.34% | 21.10% |

The final unrounded QQQ 2025 value is 21.10501065%, which displays as 21.11% under the current formatting while the classmate table displays 21.10%. This is a one-hundredth-of-a-percent display difference, not a strategy or holdings difference. Other metric discrepancies are below `1e-7`; there are no unexplained strategy-level differences.

## Backup and audit trail

The pre-alignment backup is `../DSA5205_Project2_before_classmate_alignment/`. `baseline_manifest.json` contains the original key-source hashes and the fixed data hash. `CLASSMATE_ALIGNMENT_CHANGELOG.md` records the code and documentation changes. No test-period result was used to alter the main default strategy or expand a model grid.
