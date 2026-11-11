# DSA5205 Project 2 — Submission Checklist

**Status: DRAFT, not yet a complete Canvas submission.**

## Required deliverables

| Deliverable | Status | Required action |
|---|---|---|
| Report PDF, 15–25 pages | **Missing** | Add the complete report as `Group_X_Report.pdf`. The title page must contain the team name and full roster. The supplied `Report_Working_Sections.md` contains only §4.4, Interpretation, and Reproducibility; it is not a complete report. |
| Short video, 15–25 minutes | **Missing** | Add the MP4 or provide a working, staff-accessible, downloadable link. The video must show the end-to-end run, leakage controls, model story, prediction-to-trade rule, OOS figures, and lessons learned. |
| Source code | **Ready** | `Group_X_Source_Code.zip` contains the environment file, README, exact run commands, fixed seeds, relative paths, data-build notebook, and training/evaluation notebook. |
| Contribution note | **Missing** | Add a short note describing each team member's contribution. |
| Canvas team membership | **Not verifiable here** | Confirm that all members belong to the same Canvas Project 2 group. Submit one package per team. |

Replace `Group_X` with the actual Canvas group identifier in all filenames before submission.

## Included working materials

- `Group_X_Source_Code.zip`: clean, runnable source package; no raw dataset, virtual environment, secrets, or local hard-coded directories. Includes acceptance tests and the locked environment.
- `Report_Working_Sections.md`: report-ready English text with actual computed metrics from the classmate-aligned final run.
- `CLASSMATE_ALIGNMENT_CHANGELOG.md`: code, pool, test, and output changes relative to the previous version.
- `REPRODUCIBILITY_RECORD.md`: fixed data hash, dependency versions, commands, acceptance results, and classmate-reference comparison.
- `Results/`: selected output tables and figures for cross-checking the report; main and extension results are separated.

## Code reproduction

Extract `Group_X_Source_Code.zip`, open the extracted `DSA5205_Project2` folder, and run:

```bash
conda create -n dsa5205-final python=3.11 -y
conda activate dsa5205-final
python -m pip install -r requirements.txt
python -m pytest -q
python execute_notebooks.py --notebook all
```

The acceptance tests pass with 5 tests. The data notebook requires internet access to Wikipedia and Yahoo Finance on a fresh run. It creates monthly, quarterly, and semiannual datasets. The model notebook then runs the four-model comparison and the Ridge nine-strategy extension. Outputs are written under `outputs/`; executed notebooks are saved in `notebooks/`.

## Submission deadline stated in Project2.pdf

Thursday, 12 Nov 2026, 23:59 Singapore Time (SGT, UTC+08:00).
