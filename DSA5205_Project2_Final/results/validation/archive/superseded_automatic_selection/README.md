# Superseded auxiliary artifact — do not use in the active workflow

These files are the historical output of an auxiliary `lowest_validation_mse`
rule that was added as a coding convenience during an earlier engineering
revision. It ranked the four models by validation MSE and selected LASSO
(`alpha = 0.001`).

That rule is **not** the research team's model-assessment policy. The primary
research model is Ridge, designated in
`results/validation/primary_model_contract.json` on economic and diagnostic
grounds. The archived files are retained only so that the historical run can be
audited; they are not read by the active pipeline, the notebook's final-model
decision, or the report's main results.

The archived numbers are unchanged from the original run.
