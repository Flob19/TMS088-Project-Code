"""
Independent Task 1 analysis, run from scratch without reusing the project's
cleaning rules, to sanity-check the group's existing report.  This script
prints every number the group reported + a few extras that the brief asks
for ("extract as much information as you can").
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, ccf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.cluster.hierarchy import linkage, fcluster
from pathlib import Path

ROOT = Path(__file__).parent
ASSETS = ["gurkor", "guitars", "slingshots", "stocks", "sugar", "water",
          "tranquillity"]

# ---------------------------------------------------------------------------
# 1. Raw data + outlier inspection
# ---------------------------------------------------------------------------
print("=" * 72)
print("1.  RAW DATA AND OUTLIER STRUCTURE")
print("=" * 72)
df = pd.read_csv(ROOT / "spiff_data-2.csv", index_col=0)
if "day" in df.columns:
    df = df.set_index("day")
print(f"Shape: {df.shape}  (expected 5456 x 7)")
print("Columns:", list(df.columns))

# Check for the notorious 1000-sentinel values
for c in ASSETS:
    n1000 = (df[c] == 1000).sum()
    n_nan = df[c].isna().sum()
    print(f"  {c:14s}  NaN={n_nan:3d}  ==1000: {n1000}   "
          f"min={df[c].min():.3f}  max={df[c][df[c]!=1000].max():.3f}")

# 1000-sentinels map to exactly 5 dates, shared across assets
sentinel_days = df.index[(df == 1000).any(axis=1)].tolist()
print(f"\nSentinel 1000-days: {sentinel_days}")

# ---------------------------------------------------------------------------
# 2. Cleaning
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("2.  CLEANING")
print("=" * 72)
# Replace the 1000-sentinels explicitly (cleaner than z-score)
df_clean = df.replace(1000, np.nan).copy()

# Any additional outliers after sentinel removal?
for c in ASSETS:
    x = df_clean[c].dropna()
    z = (x - x.mean()) / x.std()
    n_out = (z.abs() > 5).sum()
    print(f"  {c:14s}  after sentinel removal: |z|>5 = {n_out}")

# Gap bounds
print("\nGap positions (contiguous interior NaN blocks, excluding trailing 200):")
for c in ASSETS:
    isna = df_clean[c].isna().to_numpy()
    n = len(isna)
    # last observed day (trailing NaNs start here)
    last_obs = n - 1
    while last_obs >= 0 and isna[last_obs]:
        last_obs -= 1
    # scan for contiguous NaN block in interior
    in_block = False
    blocks = []
    for i in range(last_obs + 1):
        if isna[i] and not in_block:
            start = i; in_block = True
        elif not isna[i] and in_block:
            blocks.append((start, i - 1)); in_block = False
    # longest block
    if blocks:
        s, e = max(blocks, key=lambda r: r[1] - r[0])
        print(f"  {c:14s}  gap = day {s}..{e} (length {e-s+1})")

# ---------------------------------------------------------------------------
# 3. Returns and stationarity
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("3.  STATIONARITY (ADF)")
print("=" * 72)
rets = np.log(df_clean[ASSETS]).diff()
print(f"{'asset':14s} {'price ADF p':>12s} {'return ADF p':>14s}")
for a in ASSETS:
    pp = adfuller(df_clean[a].dropna())[1]
    rp = adfuller(rets[a].dropna())[1]
    print(f"  {a:14s} {pp:>12.4f} {rp:>14.4f}")

# ---------------------------------------------------------------------------
# 4. Four moments
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("4.  FOUR MOMENTS OF LOG-RETURNS")
print("=" * 72)
print(f"{'asset':14s} {'mean':>11s} {'vol':>11s} {'skew':>8s} {'exc.kurt':>10s}")
for a in ASSETS:
    r = rets[a].dropna().values
    print(f"  {a:14s} {r.mean():>+11.6f} {r.std(ddof=1):>11.6f} "
          f"{stats.skew(r):>+8.3f} {stats.kurtosis(r):>+10.3f}")

# ---------------------------------------------------------------------------
# 5. Autocorrelation / ARCH (Ljung-Box)
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("5.  LJUNG-BOX (lag 20)")
print("=" * 72)
print(f"{'asset':14s} {'returns p':>11s} {'squared p':>15s}")
for a in ASSETS:
    r = rets[a].dropna().values
    p1 = acorr_ljungbox(r, lags=[20], return_df=True)['lb_pvalue'].values[0]
    p2 = acorr_ljungbox(r**2, lags=[20], return_df=True)['lb_pvalue'].values[0]
    print(f"  {a:14s} {p1:>11.3g} {p2:>15.3g}")

# ---------------------------------------------------------------------------
# 6. Correlation matrix (contemporaneous log-returns)
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("6.  CORRELATIONS (log-returns)")
print("=" * 72)
C = rets.corr()
print("Ranked pairs:")
pairs = []
for i, a in enumerate(ASSETS):
    for b in ASSETS[i+1:]:
        pairs.append((a, b, C.loc[a, b]))
pairs.sort(key=lambda x: -abs(x[2]))
for a, b, r in pairs[:10]:
    print(f"  {a:12s} {b:12s} rho = {r:+.3f}")

# ---------------------------------------------------------------------------
# 7. Cross-correlations (lead-lag): something the group report skipped
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("7.  LEAD-LAG CROSS-CORRELATIONS (returns, max over h in [-5, 5])")
print("=" * 72)
print("(positive h means the second variable LEADS the first)")
print(f"{'pair':30s} {'h=0':>7s} {'best h':>8s} {'best rho':>10s}")
for i, a in enumerate(ASSETS):
    for b in ASSETS[i+1:]:
        ra = rets[a].dropna(); rb = rets[b].dropna()
        common = ra.index.intersection(rb.index)
        ra = ra.loc[common].values; rb = rb.loc[common].values
        best_h = 0; best_r = 0
        for h in range(-5, 6):
            if h == 0:
                r = np.corrcoef(ra, rb)[0, 1]
                r0 = r
            elif h > 0:
                r = np.corrcoef(ra[:-h], rb[h:])[0, 1]
            else:
                r = np.corrcoef(ra[-h:], rb[:h])[0, 1]
            if abs(r) > abs(best_r):
                best_r = r; best_h = h
        if abs(best_r) > 0.15 and abs(best_r) - abs(r0) > 0.02:
            print(f"  {a:12s} - {b:12s}   {r0:+.3f}   {best_h:+d}    {best_r:+.3f}")

# ---------------------------------------------------------------------------
# 8. Alternative grouping: hierarchical clustering on correlation distance
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("8.  HIERARCHICAL CLUSTERING on (1 - |rho|)-distance")
print("=" * 72)
dist = 1 - C.abs()
# condensed upper triangle
from scipy.spatial.distance import squareform
condensed = squareform(dist.values, checks=False)
Z = linkage(condensed, method="average")
# cut at k=3 clusters
labels = fcluster(Z, t=3, criterion="maxclust")
for a, lbl in zip(ASSETS, labels):
    print(f"  {a:14s}  cluster {lbl}")

# ---------------------------------------------------------------------------
# 9. Rolling correlation stability
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("9.  ROLLING 50-DAY CORRELATION (sign stability for top pairs)")
print("=" * 72)
for a, b in [("gurkor", "water"), ("slingshots", "guitars"),
             ("sugar", "guitars"), ("guitars", "tranquillity")]:
    roll = rets[a].rolling(50).corr(rets[b]).dropna()
    share_same_sign = (np.sign(roll) == np.sign(roll.mean())).mean()
    print(f"  {a:12s} - {b:12s}  mean={roll.mean():+.3f}  min={roll.min():+.3f}  "
          f"max={roll.max():+.3f}  share_same_sign={share_same_sign*100:.0f}%")

# ---------------------------------------------------------------------------
# 10. Return magnitudes by regime
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("10. VOLATILITY REGIMES (mean abs return in calm vs turbulent halves)")
print("=" * 72)
for a in ASSETS:
    r = rets[a].dropna()
    half = len(r) // 2
    m1 = r.iloc[:half].std(); m2 = r.iloc[half:].std()
    print(f"  {a:14s}  1st half sd = {m1*100:.2f}%  2nd half sd = {m2*100:.2f}%  "
          f"ratio = {max(m1, m2)/min(m1, m2):.2f}")
