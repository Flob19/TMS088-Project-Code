# Task 3 — Extrapolation

Files referenced by `Projekt_Short_withPACF_v2.tex` for Task 3 (200-day forecasts).

## Scripts
- `task3_extrapolation.py` — Main pipeline. Fits ARIMA-style mean models with GARCH(1,1) variance, produces 200-day forecasts, drift sensitivity, PIT calibration, and the ARIMA(1,1,1) / VAR(1) benchmark comparison.
- `task3_validation.py` — Dense rolling-origin backtest with walk-forward GARCH, drift sensitivity tables, and coverage diagnostics.

## Data
- `spiff_data-2.csv` — Raw price series.

## Output CSVs
- `task3_forecasts.csv` — Final 200-day forecasts (mean + variance per asset/horizon).
- `task3_benchmark.csv` — ARIMA(1,1,1) and VAR(1) benchmark numbers.
- `task3_backtest.csv` — Backtest aggregates.
- `task3_pit.csv` — PIT histogram data.
- `task3_validation_dense_backtest.csv` — Per-fold dense backtest output.
- `task3_validation_dense_coverage.csv` — Coverage statistics.
- `task3_validation_drift_sensitivity.csv` — Drift sensitivity sweep.

## Figures (`Pictures/`)
- `fig_garch_vs_const.png` — GARCH vs constant-variance forecasts
- `fig_forecast_paths.png` — final 200-day forecast paths
- `fig_forecast_coverage.png` — interval coverage
- `fig_dense_forecast_coverage.png` — dense backtest coverage
- `fig_pit_histograms.png` — PIT diagnostic

## Run
```
python task3_extrapolation.py
python task3_validation.py
```
