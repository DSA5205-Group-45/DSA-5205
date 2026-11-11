#!/usr/bin/env python
"""End-to-end orchestration for the Ridge-centred DSA5205 Project 2 study.

Stages
------
data        Validate the model-ready panel, write data-quality and split tables,
            optionally build the nine-strategy candidate panel.
validation  Train all four models on 2010-2019, tune on 2020-2021, write every
            validation table and the 15-figure dashboard.
designate   Freeze Ridge as the primary research model (explicit approval),
            record the policy and keep the superseded automatic-LASSO files in
            the clearly labelled archive.
refit       Refit the primary model and the three comparators on 2010-2021.
oos         Evaluate all four refitted ML models on 2022-2025.
extension   Run the preserved nine-strategy Ridge extension.
eight       Reconcile the engines and compare the eight required OOS strategies,
            producing Figures 1-6.
all         data -> validation -> designate -> refit -> oos -> extension -> eight.
            Without --approve the run stops after validation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.data_pipeline import (
    build_extension_panel,
    chronological_splits,
    data_quality_report,
    ensure_output_dirs,
    load_config,
    load_model_ready,
    resolve,
)
from src.research_policy import PrimaryDesignationNotApproved, designate_primary_model
from src.utils import write_json

EXIT_AWAITING_APPROVAL = 3


def _data_stage(cfg, args) -> None:
    ensure_output_dirs(cfg)
    df = load_model_ready(cfg, args.frequency)
    splits, summary, excluded = chronological_splits(df, cfg)
    report = data_quality_report(df, splits)
    out = resolve(cfg["paths"]["validation"])
    write_json(out / "data_quality.json", report)
    summary.to_csv(out / "split_summary.csv", index=False)
    excluded.to_csv(out / "label_embargo_excluded.csv", index=False)
    print(report)
    print(summary.to_string(index=False))
    if excluded is not None and not excluded.empty:
        print("\nCohorts removed by the realized-label embargo:")
        print(excluded.to_string(index=False))
    if not report["pass"]:
        raise RuntimeError(f"Data quality failed: {report['issues']}")

    extension_path = resolve(cfg["data"]["extension_panel"])
    if not extension_path.exists():
        if args.source_root:
            panel = build_extension_panel(cfg, Path(args.source_root), args.frequency, force=args.force)
            print(f"Built extension candidate panel: {len(panel):,} rows")
        else:
            print(
                "Extension candidate panel not found. Pass --source-root <original_pipeline> "
                "to build it, or run the extension stage later."
            )


def _validation_stage(cfg, args) -> None:
    from src.validation import run_validation
    from src.visualization import plot_validation_dashboard

    result = run_validation(cfg, frequency=args.frequency, criterion=args.criterion)
    figures = plot_validation_dashboard(cfg)
    print("\nValidation model comparison:")
    columns = [
        "model",
        "best_params",
        "validation_mse",
        "validation_rmse",
        "validation_mean_rank_ic",
        "validation_sharpe",
        "validation_cagr",
    ]
    print(result["comparison"][columns].to_string(index=False))
    print(f"\nGenerated {len(figures)} validation figures in {resolve(cfg['paths']['validation_figures'])}")


def _designate_stage(cfg, args) -> dict:
    from src.research_policy import designation_table

    contract = designate_primary_model(cfg, approve=args.approve, note=args.note)
    print("\nPrimary-model designation")
    print(designation_table(cfg)[["model", "research_role", "validation_mse", "validation_mean_rank_ic", "validation_sharpe", "is_primary"]].to_string(index=False))
    print(
        f"\nFrozen primary model: {contract['primary_model']} "
        f"{contract['selected_hyperparameters']} ({contract['hyperparameter_rule']})"
    )
    print(f"Contract: {resolve(cfg['paths']['validation']) / 'primary_model_contract.json'}")
    return contract


def _refit_stage(cfg, args, contract) -> None:
    from src.final_refit import refit_models

    result = refit_models(cfg, frequency=args.frequency, contract=contract)
    print("\nFinal refit summary:")
    print(result["summary"][["model", "is_selected", "hyperparameters", "n_training_observations"]].to_string(index=False))


def _oos_stage(cfg, args) -> None:
    from src.oos_evaluation import run_oos
    from src.visualization import plot_oos_dashboard

    result = run_oos(cfg, frequency=args.frequency)
    figures = plot_oos_dashboard(cfg)
    columns = ["model", "mse", "oos_sharpe", "oos_cagr", "oos_max_drawdown"]
    print("\nOut-of-sample comparison (descriptive; the primary designation is already frozen):")
    print(result["comparison"][columns].to_string(index=False))
    print(f"\nGenerated {len(figures)} OOS figures in {resolve(cfg['paths']['oos_figures'])}")


def _extension_stage(cfg, args) -> None:
    from src.ridge_extension import run_ridge_extension
    from src.reconciliation import run_ridge_r4_reconciliation

    result = run_ridge_extension(cfg, frequency=args.frequency)
    print("\nNine-strategy Ridge extension:")
    print(result["summary"][["strategy", "annualized_return", "sharpe_rf0", "max_drawdown"]].to_string(index=False))
    reconciliation = run_ridge_r4_reconciliation(cfg)
    print(
        "\nMain Ridge and the extension's internal Ridge estimator: same rule, different fit; "
        f"mean holdings overlap {reconciliation['comparison']['mean_holdings_overlap']:.1%}, "
        f"CAGR difference {reconciliation['comparison']['cagr_difference_r4_minus_main'] * 100:+.2f} pp"
    )


def _score_stage(cfg, args) -> None:
    from src.score_position import run_score_position_example

    result = run_score_position_example(cfg)
    print(f"\nScore-to-position example ({result['decision_date']}):")
    print(result["table"][["rank", "ticker", "prediction", "target_weight"]].head(12).to_string(index=False))


def _eight_stage(cfg, args) -> None:
    from src.oos_comparison import run_eight_strategy_comparison

    result = run_eight_strategy_comparison(cfg, frequency=args.frequency)
    print("\nEight-strategy OOS comparison:")
    print(
        result["metrics"][
            ["strategy", "role", "annualized_return", "annualized_volatility", "sharpe_ratio", "max_drawdown", "average_turnover"]
        ].to_string(index=False)
    )
    print(f"\nAlignment checks: {result['checks']}")
    print(f"Figures 1-6 written to {resolve(cfg['paths']['oos_figures'])}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--stage",
        choices=["data", "validation", "designate", "refit", "oos", "extension", "eight", "score", "all"],
        default="all",
    )
    parser.add_argument("--frequency", default=None, choices=["monthly", "quarterly", "semiannual", None])
    parser.add_argument(
        "--criterion",
        default=None,
        choices=["lowest_validation_mse", "highest_validation_sharpe", None],
        help="Hyperparameter ranking criterion used during validation.",
    )
    parser.add_argument("--approve", action="store_true", help="Explicitly approve the primary-model designation.")
    parser.add_argument("--note", default="", help="Free-text approval note recorded in the contract.")
    parser.add_argument("--source-root", default=None, help="Path to the original pipeline folder (for --stage data).")
    parser.add_argument("--force", action="store_true", help="Rebuild derived artefacts.")
    args = parser.parse_args(argv)

    cfg = load_config()
    args.frequency = args.frequency or cfg["project"]["frequency"]
    args.criterion = args.criterion or cfg["selection"]["criterion"]

    if args.stage == "data":
        _data_stage(cfg, args)
        return 0
    if args.stage == "validation":
        _data_stage(cfg, args)
        _validation_stage(cfg, args)
        return 0
    if args.stage == "designate":
        try:
            _designate_stage(cfg, args)
        except PrimaryDesignationNotApproved as exc:
            print(f"STOP: {exc}")
            return EXIT_AWAITING_APPROVAL
        return 0
    if args.stage == "refit":
        from src.research_policy import active_contract

        _refit_stage(cfg, args, active_contract(cfg))
        return 0
    if args.stage == "oos":
        _oos_stage(cfg, args)
        return 0
    if args.stage == "extension":
        _extension_stage(cfg, args)
        return 0
    if args.stage == "eight":
        _eight_stage(cfg, args)
        return 0
    if args.stage == "score":
        _score_stage(cfg, args)
        return 0

    # ---- all ----------------------------------------------------------------
    _data_stage(cfg, args)
    _validation_stage(cfg, args)
    if not args.approve:
        print(
            "\nSTOP: validation is complete. The test period has NOT been opened.\n"
            "Inspect the validation comparison and re-run with:\n"
            "  python run_pipeline.py --stage all --approve"
        )
        return EXIT_AWAITING_APPROVAL
    try:
        contract = _designate_stage(cfg, args)
    except PrimaryDesignationNotApproved as exc:
        print(f"STOP: {exc}")
        return EXIT_AWAITING_APPROVAL
    _refit_stage(cfg, args, contract)
    _oos_stage(cfg, args)
    _score_stage(cfg, args)
    _extension_stage(cfg, args)
    _eight_stage(cfg, args)
    print("\nPipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
