"""
Task 1 — figure generation.

Recreates every figure referenced in the LaTeX report (Projekt_Short_withPACF_v2.tex)
for Section "Task 1. Data analysis". Run from inside the Task 1 folder
(the script uses Path(__file__).parent so 'spiff_data-2.csv' and 'Pictures/'
must sit next to it).

Outputs written to ./Pictures/ :
    timeseries_plot.jpg
    fig1_price_series.png
    fig2_log_returns.png
    fig14_normalised_prices.png
    fig6_distributions.png
    acf_returns_v2.png
    acf_sqreturns_v2.png
    pacf_returns_v2.png
    pacf_sqreturns_v2.png
    Correlation matrix of log-returns.jpg
    fig_corr_sq.png
    fig_rolling_corr.png
    fig_dendrogram.png
    kmeans_vol_vs_kurtosis.jpg

Cleaning rules and parameters are identical to task1_independent.py.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from statsmodels.tsa.stattools import acf, pacf
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

np.random.seed(0)

ROOT = Path(__file__).parent
PIC = ROOT / "Pictures"
PIC.mkdir(exist_ok=True)

ASSETS = ["gurkor", "guitars", "slingshots", "stocks",
          "sugar", "water", "tranquillity"]
COLORS = {
    "gurkor":       "#1f77b4",  # blue
    "guitars":      "#ff7f0e",  # orange
    "slingshots":   "#2ca02c",  # green
    "stocks":       "#d62728",  # red
    "sugar":        "#9467bd",  # purple
    "water":        "#8c564b",  # brown
    "tranquillity": "#e377c2",  # pink
}

GAP_LEN = 50          # interior gap to be interpolated in Task 2
FORECAST_LEN = 200    # trailing block to be forecast in Task 3

# ---------------------------------------------------------------------------
# 1. Load and clean
# ---------------------------------------------------------------------------
raw = pd.read_csv(ROOT / "spiff_data-2.csv", index_col=0)
if "day" in raw.columns:
    raw = raw.set_index("day")
raw = raw[ASSETS]  # enforce column order

# 1000-sentinels -> NaN
df = raw.replace(1000, np.nan).copy()
# Z>8 -> NaN (per series, after removing sentinels)
for c in ASSETS:
    s = df[c]
    mu, sd = s.mean(), s.std()
    df.loc[(np.abs((s - mu) / sd) > 8), c] = np.nan

# Identify per-asset 50-day interior gap and 200-day trailing forecast region.
gaps = {}
fore = {}
for c in ASSETS:
    nan_mask = df[c].isna().to_numpy()
    # trailing 200 days
    fore_start = len(df) - FORECAST_LEN
    fore[c] = (fore_start, len(df))
    # interior gap: search a contiguous NaN run of length GAP_LEN strictly
    # before the trailing region.
    run_start = None
    for i in range(fore_start):
        if nan_mask[i]:
            if run_start is None:
                run_start = i
            if i - run_start + 1 == GAP_LEN:
                gaps[c] = (run_start, run_start + GAP_LEN)
                break
        else:
            run_start = None

# Log-returns on the cleaned series
log_ret = np.log(df).diff()


# ---------------------------------------------------------------------------
# Figure: timeseries_plot.jpg  (raw with 1000-spikes, simple legend)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 8))
for c in ASSETS:
    ax.plot(raw.index, raw[c], label=c, color=COLORS[c], linewidth=0.8)
ax.set_title("Time Series", fontsize=16)
ax.set_xlabel("Day"); ax.set_ylabel("Value")
ax.legend(loc="center right", fontsize=11)
plt.tight_layout()
plt.savefig(PIC / "timeseries_plot.jpg", dpi=150, bbox_inches="tight")
plt.close()


# ---------------------------------------------------------------------------
# Figure: fig1_price_series.png   (cleaned prices, 7 panels, gap/forecast shading)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(len(ASSETS), 1, figsize=(11, 12), sharex=True)
for ax, c in zip(axes, ASSETS):
    ax.plot(df.index, df[c], color=COLORS[c], linewidth=0.6)
    g0, g1 = gaps[c]
    f0, f1 = fore[c]
    ax.axvspan(g0, g1, color="gold", alpha=0.55,
               label="Interpolation gap (Task 2)")
    ax.axvspan(f0, f1, color="red", alpha=0.18,
               label="Forecast region (Task 3)")
    ax.set_ylabel(c, fontsize=9)
    ax.grid(alpha=0.25)
axes[0].set_title("Price Series (after outlier removal)",
                  fontsize=14, fontweight="bold")
axes[-1].set_xlabel("Day")
# single legend at top
handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles, labels, loc="upper left", fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(PIC / "fig1_price_series.png", dpi=140, bbox_inches="tight")
plt.close()


# ---------------------------------------------------------------------------
# Figure: fig2_log_returns.png    (same layout but log-returns)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(len(ASSETS), 1, figsize=(11, 12), sharex=True)
for ax, c in zip(axes, ASSETS):
    ax.plot(log_ret.index, log_ret[c], color=COLORS[c], linewidth=0.5)
    g0, g1 = gaps[c]
    ax.axvspan(g0, g1, color="gold", alpha=0.55)
    f0, f1 = fore[c]
    ax.axvspan(f0, f1, color="red", alpha=0.18)
    ax.set_ylabel(c, fontsize=9)
    ax.grid(alpha=0.25)
axes[0].set_title("Log-Return Series", fontsize=14, fontweight="bold")
axes[-1].set_xlabel("Day")
plt.tight_layout()
plt.savefig(PIC / "fig2_log_returns.png", dpi=140, bbox_inches="tight")
plt.close()


# ---------------------------------------------------------------------------
# Figure: fig14_normalised_prices.png  (all 7 indexed to 100)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 4))
for c in ASSETS:
    s = df[c].dropna()
    norm = 100.0 * s / s.iloc[0]
    ax.plot(norm.index, norm.values, label=c, color=COLORS[c], linewidth=0.8)
ax.axhline(100, color="black", linestyle="--", linewidth=0.6)
ax.set_title("Normalised Price Index  (base = 100 at first observation)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Day"); ax.set_ylabel("Normalised Price (base = 100)")
ax.legend(ncol=4, fontsize=9, loc="upper right")
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(PIC / "fig14_normalised_prices.png",
            dpi=140, bbox_inches="tight")
plt.close()


# ---------------------------------------------------------------------------
# Figure: fig6_distributions.png  (hist + QQ per asset)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(len(ASSETS), 2, figsize=(11, 16))
for i, c in enumerate(ASSETS):
    r = log_ret[c].dropna().values
    mu, sd = r.mean(), r.std()
    skew = stats.skew(r); kurt = stats.kurtosis(r)  # excess

    # left: histogram with fitted normal
    axL = axes[i, 0]
    axL.hist(r, bins=80, density=True, color=COLORS[c],
             alpha=0.75, edgecolor="white", linewidth=0.2)
    xx = np.linspace(r.min(), r.max(), 400)
    axL.plot(xx, stats.norm.pdf(xx, mu, sd), "k--",
             linewidth=1.0, label="Normal fit")
    axL.set_title(c, fontsize=9); axL.set_xlabel("Log-return")
    axL.set_ylabel("Density")
    axL.text(0.02, 0.92, f"Kurt={kurt:.2f}\nSkew={skew:.2f}",
             transform=axL.transAxes, fontsize=7,
             verticalalignment="top")
    axL.legend(fontsize=7, loc="upper left")

    # right: QQ-plot
    axR = axes[i, 1]
    stats.probplot(r, dist="norm", plot=axR)
    axR.get_lines()[0].set_color(COLORS[c])
    axR.get_lines()[0].set_markersize(2)
    axR.get_lines()[1].set_color("black")
    axR.get_lines()[1].set_linestyle("--")
    axR.set_title(f"{c} - QQ-Plot", fontsize=9)
    axR.set_xlabel("Theoretical quantiles")
    axR.set_ylabel("Sample quantiles")

fig.suptitle("Return Distributions - Histogram & QQ-Plot",
             fontsize=14, fontweight="bold", y=1.00)
plt.tight_layout()
plt.savefig(PIC / "fig6_distributions.png", dpi=130, bbox_inches="tight")
plt.close()


# ---------------------------------------------------------------------------
# Helper: 4x2 grid of (P)ACF for the seven assets, with 95% bounds.
# ---------------------------------------------------------------------------
def _grid_corr_plot(series_dict, fname, title, nlags=50, squared=False):
    fig, axes = plt.subplots(4, 2, figsize=(11, 9))
    axes = axes.flatten()
    for i, c in enumerate(ASSETS):
        ax = axes[i]
        x = series_dict[c]
        n = len(x)
        vals = series_dict["_func"](x, nlags=nlags)
        # drop lag 0
        vals = vals[1:]
        ax.bar(range(1, nlags + 1), vals,
               color=COLORS[c], width=0.15)
        ax.scatter(range(1, nlags + 1), vals, color=COLORS[c], s=10)
        ax.axhline(0, color="grey", linewidth=0.5)
        ci = 1.96 / np.sqrt(n)
        ax.axhline(+ci, color="grey", linestyle="--", linewidth=0.5)
        ax.axhline(-ci, color="grey", linestyle="--", linewidth=0.5)
        ax.set_title(c, fontsize=9)
        ax.set_xlabel("Lag"); ax.set_ylabel(r"$\rho(h)$" if not squared else r"$\rho(h)$ for $r^2$")
        ax.grid(alpha=0.25)
    axes[-1].axis("off")  # 8th panel empty (we have 7 assets)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PIC / fname, dpi=140, bbox_inches="tight")
    plt.close()


# ACF / PACF of returns and squared returns
ret_dict = {c: log_ret[c].dropna().values for c in ASSETS}
ret2_dict = {c: (log_ret[c].dropna().values ** 2) for c in ASSETS}

ret_dict["_func"] = lambda x, nlags: acf(x, nlags=nlags, fft=False)
_grid_corr_plot(ret_dict, "acf_returns_v2.png",
                "ACF of log-returns (lag 0 omitted)", nlags=50, squared=False)

ret_dict["_func"] = lambda x, nlags: pacf(x, nlags=nlags, method="ywm")
_grid_corr_plot(ret_dict, "pacf_returns_v2.png",
                "PACF of log-returns (lag 0 omitted)", nlags=50, squared=False)

ret2_dict["_func"] = lambda x, nlags: acf(x, nlags=nlags, fft=False)
_grid_corr_plot(ret2_dict, "acf_sqreturns_v2.png",
                "ACF of squared log-returns "
                "(volatility clustering / ARCH effects, lag 0 omitted)",
                nlags=50, squared=True)

ret2_dict["_func"] = lambda x, nlags: pacf(x, nlags=nlags, method="ywm")
_grid_corr_plot(ret2_dict, "pacf_sqreturns_v2.png",
                "PACF of squared log-returns (lag 0 omitted)",
                nlags=50, squared=True)


# ---------------------------------------------------------------------------
# Correlation heatmaps (log-returns and squared log-returns)
# ---------------------------------------------------------------------------
def _heatmap(corr, title, fname, cmap="RdBu_r"):
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr.values, cmap=cmap, vmin=-corr.values.min().__abs__() if False else -0.25, vmax=1.0)
    # use symmetric bounds for the log-return matrix, fixed for sq
    if "squared" in title:
        im = ax.imshow(corr.values, cmap=cmap, vmin=-0.2, vmax=1.0)
    else:
        im = ax.imshow(corr.values, cmap=cmap, vmin=-0.3, vmax=1.0)
    ax.set_xticks(range(len(ASSETS))); ax.set_yticks(range(len(ASSETS)))
    ax.set_xticklabels(ASSETS, fontsize=11)
    ax.set_yticklabels(ASSETS, fontsize=11)
    for i in range(len(ASSETS)):
        for j in range(len(ASSETS)):
            v = corr.values[i, j]
            color = "white" if abs(v) > 0.7 else "black"
            ax.text(j, i, f"{v:.2g}" if abs(v) < 1 else "1",
                    ha="center", va="center", color=color, fontsize=11)
    ax.set_title(title, fontsize=14)
    fig.colorbar(im, ax=ax, shrink=0.85)
    plt.tight_layout()
    plt.savefig(PIC / fname, dpi=150, bbox_inches="tight")
    plt.close()

corr_r = log_ret[ASSETS].corr()
_heatmap(corr_r, "Correlation matrix of log-returns",
         "Correlation matrix of log-returns.jpg")

corr_r2 = (log_ret[ASSETS] ** 2).corr()
_heatmap(corr_r2, "Correlation matrix of squared log-returns $r_t^2$",
         "fig_corr_sq.png")


# ---------------------------------------------------------------------------
# Rolling 50-day correlation for the two strong peer pairs
# ---------------------------------------------------------------------------
def _rolling_corr_panel(ax, a, b):
    rc = log_ret[a].rolling(50).corr(log_ret[b])
    m = rc.mean()
    ax.plot(rc.index, rc.values, color=COLORS[a], linewidth=0.8)
    ax.axhline(m, color="red", linestyle="--", linewidth=1.0,
               label=f"mean = {m:+.2f}")
    ax.axhline(0, color="black", linewidth=0.4)
    ax.set_title(f"{a} vs {b}", fontsize=11)
    ax.set_xlabel("day"); ax.set_ylabel(r"$\rho_{50d}$")
    ax.grid(alpha=0.25); ax.legend(loc="upper left", fontsize=8)

fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
_rolling_corr_panel(axes[0], "gurkor", "water")
_rolling_corr_panel(axes[1], "slingshots", "guitars")
plt.tight_layout()
plt.savefig(PIC / "fig_rolling_corr.png", dpi=140, bbox_inches="tight")
plt.close()


# ---------------------------------------------------------------------------
# Hierarchical dendrogram on d_ij = 1 - |rho|, average linkage
# ---------------------------------------------------------------------------
dist = 1.0 - corr_r.abs()
condensed = []
n = len(ASSETS)
for i in range(n):
    for j in range(i + 1, n):
        condensed.append(dist.values[i, j])
Z = linkage(condensed, method="average")
fig, ax = plt.subplots(figsize=(11, 5))
dendrogram(Z, labels=ASSETS, ax=ax, color_threshold=0.9)
ax.set_title("Hierarchical clustering of log-return series (average linkage)",
             fontsize=12)
ax.set_ylabel(r"distance $= 1 - |\rho|$")
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(PIC / "fig_dendrogram.png", dpi=140, bbox_inches="tight")
plt.close()


# ---------------------------------------------------------------------------
# K-Means clustering on (volatility, excess kurtosis)
# ---------------------------------------------------------------------------
vol = log_ret.std().reindex(ASSETS).values
kur = log_ret.apply(lambda s: stats.kurtosis(s.dropna())).reindex(ASSETS).values
feat = np.column_stack([vol, kur])
scaler = StandardScaler()
feat_s = scaler.fit_transform(feat)
km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(feat_s)
labels = km.labels_

# Re-label so that:
#   Cluster 0 = lowest mean volatility   (gurkor / water style)
#   Cluster 1 = highest mean kurtosis    (sugar / slingshots tail-risk)
#   Cluster 2 = the remaining cluster    (stocks / tranquillity)
# This matches the cluster numbering used in the LaTeX report so that
# Cluster 0/1/2 colours line up regardless of K-Means random init.
mean_vol = np.array([vol[labels == k].mean() for k in range(3)])
mean_kur = np.array([kur[labels == k].mean() for k in range(3)])
low_vol = int(np.argmin(mean_vol))
high_kur = int(np.argmax(np.where(np.arange(3) == low_vol, -np.inf, mean_kur)))
rest = [k for k in range(3) if k not in (low_vol, high_kur)][0]
remap = {low_vol: 0, high_kur: 1, rest: 2}
labels = np.array([remap[k] for k in labels])

fig, ax = plt.subplots(figsize=(10, 7))
cluster_colors = {0: "#1f77b4", 1: "#d62728", 2: "#2ca02c"}
for ci in sorted(set(labels)):
    mask = labels == ci
    ax.scatter(vol[mask], kur[mask],
               c=cluster_colors[ci], s=130, edgecolor="black",
               linewidth=1.0, label=f"Cluster {ci}", zorder=3)
for i, a in enumerate(ASSETS):
    ax.annotate(a, (vol[i], kur[i]),
                xytext=(8, 4), textcoords="offset points",
                fontsize=11, fontweight="bold")
ax.set_title("Asset Grouping: Volatility vs. Kurtosis Risk Profile",
             fontsize=15)
ax.set_xlabel("Volatility (Daily Standard Deviation)", fontsize=12)
ax.set_ylabel("Kurtosis (Fat-Tail / Black Swan Risk)", fontsize=12)
ax.legend(title="Statistical Clusters", fontsize=11, title_fontsize=12,
          loc="upper left")
ax.grid(alpha=0.4, linestyle="--")
plt.tight_layout()
plt.savefig(PIC / "kmeans_vol_vs_kurtosis.jpg",
            dpi=140, bbox_inches="tight")
plt.close()


print("All Task 1 figures regenerated in", PIC)
