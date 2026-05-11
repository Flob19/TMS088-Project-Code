# Task 2 — Interpolation

Files referenced by `Projekt_Short_withPACF_v2.tex` for Task 2 (gap interpolation).

## Scripts
- `task2_interpolation.py` — Main pipeline. Builds the univariate Brownian-bridge baseline and the bivariate-bridge-with-peer estimator, handles heteroscedasticity with GARCH(1,1), runs the synthetic-gap validation, and produces all final interpolated values.
- `task2_extra_figures.py` — Generates the supporting illustrative figures used in the report.

## Data
- `spiff_data-2.csv` — Raw price series.

## Output CSVs
- `task2_results.csv` — Per-method validation results.
- `task2_summary.csv` — Gap-midpoint summary.
- `task2_interpolated_values.csv` — Final per-day estimates for each missing point.
- `task2_leadlag.csv` — Lead-lag scan used for peer selection.

## Figures (`Pictures/`)
- `fig_interp_compare.png` — comparison of interpolation methods
- `fig_target_vs_peer.png` — target vs chosen peer series
- `fig_garch_variance.png` — GARCH variance inside the gap
- `fig_synth_fold_example.png` — synthetic-gap validation fold
- `fig_interp_paths.png` — final interpolation paths
- `fig_interp_uncertainty.png` — uncertainty bands

## Run
```
python task2_interpolation.py
python task2_extra_figures.py
```
