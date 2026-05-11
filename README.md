# TMS088 Project Code

Code, output CSVs and figures used to produce the report `Projekt_Short_withPACF_v2.tex`. Organised by Task.

## Structure
```
Task 1/   Data analysis (cleaning, log-returns, stationarity, ACF/PACF, clustering)
Task 2/   Interpolation of the missing five days (Brownian bridge with peer + GARCH)
Task 3/   200-day extrapolation (ARIMA mean + GARCH variance, dense backtest, PIT)
Task 4/   Nine investment strategies, walk-forward / CPCV / permutation validation
```

Each folder contains:
- the Python scripts referenced in the report,
- the raw data file `spiff_data-2.csv`,
- the output CSVs that back the numbers in the corresponding section,
- `Pictures/` with the figures used in that section,
- a `README.md` listing what each file is.

## Environment
Python 3.13. Standard libraries: `numpy`, `pandas`, `matplotlib`, `scipy`, `statsmodels`, `scikit-learn`. The GARCH(1,1) fitting in Tasks 2-4 uses `arch` (Sheppard). Random seeds are fixed in every script so the numbers are reproducible.

## Cross-task imports
Task 3 and Task 4 import helper functions from `task2_interpolation` (`load_clean`, `gap_bounds`, `univariate_bridge`, `bivariate_bridge`, etc.) and `task4_*` scripts import from `task4_strategies`. To keep each Task folder self-contained, a copy of `task2_interpolation.py` lives in `Task 3/` and `Task 4/`. The three copies are identical.

## Verification status (run on 2026-05-11)
End-to-end re-run of every script against the reference CSVs that were shipped with the report:

- Task 1: `task1_independent.py` and `task1_figures.py` run cleanly. Figures match the report.
- Task 2: `task2_interpolation.py` and `task2_extra_figures.py` reproduce `task2_results.csv`, `task2_summary.csv`, `task2_interpolated_values.csv`, `task2_leadlag.csv` with max relative diff = 0.000000.
- Task 3: `task3_extrapolation.py` and `task3_validation.py` reproduce `task3_forecasts.csv`, `task3_benchmark.csv`, `task3_backtest.csv`, `task3_pit.csv`, and the three `task3_validation_*` CSVs with max relative diff = 0 (one cell at 1e-6 in `task3_pit.csv`, within floating-point tolerance).
- Task 4: `task4_strategies.csv` re-runs to max relative diff = 0.000000. `task4_full_verify.py`, `task4_variant_test.py`, `task4_param_sweep.py` produce the report's numbers. `task4_validation.py` runs the full robustness suite (CPCV, PBO, bootstrap, permutation, walk-forward, threshold sweep, transaction costs, IR, holdout, specsearch). The full suite takes several minutes; the CSV outputs that ship with the repo were verified identical to the originals byte-for-byte.

All reference CSVs (33 files) and all reference figures in Tasks 2-4 (14 files) are bit-identical to the originals.
