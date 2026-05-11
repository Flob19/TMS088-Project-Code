# Task 4 — Investment strategies

Files referenced by `Projekt_Short_withPACF_v2.tex` for Task 4 (nine strategies, evaluation, lead-lag robustness).

## Scripts
- `task4_strategies.py` — Main pipeline. Implements the nine strategies, produces the equity curves, drawdowns, Sharpe splits, and the momentum grid.
- `task4_validation.py` — Full validation suite: holdout, walk-forward, specification search, CPCV + PBO, permutation tests, bootstrap, threshold sweep, transaction costs, extreme-event check, information ratio.
- `task4_param_sweep.py` — Parameter sweep used to confirm robustness of the chosen settings.
- `task4_full_verify.py` — End-to-end verification run.
- `task4_leadlag_variants.py` — Variants used to test the robustness of the lead-lag sugar finding.
- `task4_variant_test.py` — Variant test runner.

## Data / inputs
- `spiff_data-2.csv` — Raw price series.
- `task3_forecasts.csv` — Forecast means and variances from Task 3, used as an input to forecast-driven strategies.

## Output CSVs
- `task4_strategies.csv` — Final performance table for all nine strategies.
- `task4_momentum_grid.csv` — Momentum lookback / holding grid.
- `task4_validation_holdout.csv`, `task4_validation_walkforward.csv`, `task4_validation_walkforward_wide.csv` — out-of-sample evaluation.
- `task4_validation_specsearch.csv` — specification search.
- `task4_validation_cpcv.csv`, `task4_validation_pbo_summary.csv` — combinatorial purged CV + PBO.
- `task4_validation_permutation.csv`, `task4_validation_permutation_dist.csv` — permutation tests.
- `task4_validation_bootstrap.csv`, `task4_validation_bootstrap_dist.csv` — bootstrap distributions.
- `task4_validation_threshold.csv` — threshold sweep.
- `task4_validation_transaction_costs.csv` — cost-sensitivity table.
- `task4_validation_extreme_events.csv` — extreme-event diagnostic.
- `task4_validation_information_ratio.csv` — IR summary.

## Figures (`Pictures/`)
- `fig_strategy_equity.png` — equity curves
- `fig_strategy_drawdowns.png` — drawdowns
- `fig_strategy_sharpe_split.png` — Sharpe split (in/out of sample)

## Run
```
python task4_strategies.py
python task4_validation.py
```
