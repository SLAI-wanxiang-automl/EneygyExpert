# Validation — 2026-10-10

Two independent code-review subagents examined the alias collision and data/rolling/tuning boundaries.

Fixes:
- MedianEnsembleAdapter defaults to MedianModelEnsemble, avoiding reserved prediction-ensemble result names. Regression verifies that candidate and generated ensemble results coexist.
- ForecastData rejects nonnumeric and nonfinite targets, preventing invalid targets from contaminating evaluation.

Local validation (/opt/anaconda3/bin/python):
- data/window/backtesting/tuning/test_rolling/pipeline/native_automl: 29 passed.
- rolling_model_adapters excluding real Prophet: 4 passed, 31 skipped, 1 deselected (missing optional backends).
- Originally reported native StatsForecast integration case skipped because statsforecast is unavailable in this environment.
- Real Prophet test crashes in the local cmdstanpy native dependency; full backend verification is incomplete.
- Synthetic examples/quickstart.py completed and exported reports.

Publication excludes local datasets, experiment outputs, copied virtual environments and secrets. Original checkout has corrupt Git objects; publication uses a new source snapshot and preserves upstream attribution. Full legacy history is not recovered.
