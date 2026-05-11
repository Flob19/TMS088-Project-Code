# Task 1 — Data analysis

Files referenced by `Projekt_Short_withPACF_v2.tex` for Task 1 (Data analysis).

## Scripts
- `task1_independent.py` — Standalone sanity-check script. Prints every number in Task 1 (cleaning, log-returns, stationarity tests, distribution moments, ACF/PACF, correlations, K-Means clustering, hierarchical dendrogram). Produces no figures.
- `task1_figures.py` — Regenerates every figure referenced in Section 1 of the report. Same cleaning rules and seeds as `task1_independent.py`. Writes to `Pictures/`.
- `Felix F Data analys.ipynb` — Original working notebook. Kept for reference. The figures used in the report are now reproduced by `task1_figures.py`.

## Data
- `spiff_data-2.csv` — Raw price series (7 assets, 5456 observations).

## Figures (`Pictures/`)
Referenced in the LaTeX report:
- `timeseries_plot.jpg` — overview of all raw series
- `fig1_price_series.png`, `fig2_log_returns.png` — price and log-return panels
- `acf_returns_v2.png`, `acf_sqreturns_v2.png` — ACF of returns and squared returns
- `pacf_returns_v2.png`, `pacf_sqreturns_v2.png` — PACF of returns and squared returns
- `fig6_distributions.png` — return distributions
- `fig14_normalised_prices.png` — normalised prices
- `Correlation matrix of log-returns.jpg` — correlation heatmap
- `fig_rolling_corr.png` — rolling correlation
- `fig_corr_sq.png` — correlation of squared returns
- `kmeans_vol_vs_kurtosis.jpg` — K-Means clustering on volatility vs kurtosis
- `fig_dendrogram.png` — hierarchical clustering dendrogram

## Run
```
python task1_independent.py   # prints the numerical tables
python task1_figures.py       # writes the figures into Pictures/
```

## Notes on reproduction
- Cleaning rule: drop the five shared 1000-sentinel days and any per-asset $|z|>8$ outlier (matches `task1_independent.py`).
- K-Means: **open question to confirm with the group.** The LaTeX report text says K-Means uses the standardised (volatility, kurtosis) feature pair, but that combination puts guitars in the same cluster as slingshots/sugar (their vol+kurt coordinates are nearly identical), not with gurkor/water as the published figure shows. To reproduce the published figure exactly, `task1_figures.py` runs K-Means on all four standardised moments (mean, std, skew, kurt) while keeping (std, kurt) as the scatter axes. See the comment block in `task1_figures.py` for context. Worth checking what the original setup was before final submission.
- All seeds are fixed (`np.random.seed(0)` and `KMeans(random_state=0)`).
