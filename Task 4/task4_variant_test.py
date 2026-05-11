"""Quick test: combine 100% all-in with vol-scaled size for lead-lag sugar.

Variants tested:
  A) 1/7 binary (baseline, in report)
  B) 100% all-in binary (in report Tabell 24)
  D) 1/7 + vol-scaled (proposed addition to report)
  D') 100% + vol-scaled (the question asked)
"""
import numpy as np
import pandas as pd
from task4_strategies import build_price_panel, ASSETS, RF_DAILY, RF_ANNUAL, DAYS_PER_YEAR

def sharpe(r):
    r = pd.Series(r).dropna()
    excess = r - RF_DAILY
    return np.sqrt(DAYS_PER_YEAR) * excess.mean() / excess.std()

def max_dd(r):
    r = pd.Series(r).dropna()
    eq = (1 + r).cumprod()
    return (eq / eq.cummax() - 1).min()

def terminal(r):
    r = pd.Series(r).dropna()
    return float((1 + r).cumprod().iloc[-1])

def lo_ci(r, alpha=0.05):
    """Lo (2002) Sharpe CI under iid Gaussian assumption."""
    r = pd.Series(r).dropna()
    n = len(r)
    sr = sharpe(r) / np.sqrt(DAYS_PER_YEAR)  # daily Sharpe
    se = np.sqrt((1 + 0.5 * sr**2) / n)
    z = 1.96
    sr_ann = sr * np.sqrt(DAYS_PER_YEAR)
    se_ann = se * np.sqrt(DAYS_PER_YEAR)
    return sr_ann - z*se_ann, sr_ann + z*se_ann

def variant_A_binary_eqw(prices):
    """1/7 binary -- baseline already in report."""
    r = np.log(prices).diff()
    signal = 0.5 * (r["guitars"].shift(1) + r["slingshots"].shift(1))
    w = pd.DataFrame(0.0, index=prices.index, columns=ASSETS)
    for a in ASSETS:
        if a != "sugar":
            w[a] = 1.0 / 7
    w["sugar"] = (signal > 0).astype(float) * (1.0 / 7)
    port_r = (w * r).sum(axis=1)
    cash_w = 1.0 - w.sum(axis=1)
    return (port_r + cash_w * RF_DAILY).dropna()

def variant_B_full(prices):
    """100% all-in sugar when signal > 0, else cash."""
    r = np.log(prices).diff()
    signal = 0.5 * (r["guitars"].shift(1) + r["slingshots"].shift(1))
    w_sugar = (signal > 0).astype(float)
    port_r = w_sugar * r["sugar"]
    cash_w = 1.0 - w_sugar
    return (port_r + cash_w * RF_DAILY).dropna()

def variant_D_eqw_vol(prices, target_vol_ann=0.30, vol_window=20):
    """1/7 binary x vol-scaling on sugar position only.

    Sugar weight: (signal>0) * 1/7 * (target / sigma_sugar_t).
    Capped at 1/7 max (don't lever beyond baseline).
    Other 6: 1/7 each.
    """
    r = np.log(prices).diff()
    signal = 0.5 * (r["guitars"].shift(1) + r["slingshots"].shift(1))
    sigma_sugar = r["sugar"].rolling(vol_window).std() * np.sqrt(DAYS_PER_YEAR)
    target_daily_to_ann = 1.0  # already annualised
    scale = (target_vol_ann / sigma_sugar).clip(upper=1.0).shift(1)  # use yesterday's vol
    w = pd.DataFrame(0.0, index=prices.index, columns=ASSETS)
    for a in ASSETS:
        if a != "sugar":
            w[a] = 1.0 / 7
    w["sugar"] = (signal > 0).astype(float) * (1.0 / 7) * scale
    port_r = (w * r).sum(axis=1)
    cash_w = 1.0 - w.sum(axis=1)
    return (port_r + cash_w * RF_DAILY).dropna()

def variant_Dp_full_vol(prices, target_vol_ann=0.30, vol_window=20):
    """100% binary x vol-scaling on sugar -- the user's question.

    Sugar weight: (signal>0) * (target / sigma_sugar_t), capped at 1.0.
    Otherwise cash.
    """
    r = np.log(prices).diff()
    signal = 0.5 * (r["guitars"].shift(1) + r["slingshots"].shift(1))
    sigma_sugar = r["sugar"].rolling(vol_window).std() * np.sqrt(DAYS_PER_YEAR)
    scale = (target_vol_ann / sigma_sugar).clip(upper=1.0).shift(1)
    w_sugar = (signal > 0).astype(float) * scale
    port_r = w_sugar * r["sugar"]
    cash_w = 1.0 - w_sugar
    return (port_r + cash_w * RF_DAILY).dropna()

def variant_Dp_uncapped(prices, target_vol_ann=0.30, vol_window=20):
    """Same as D' but without the cap (full risk targeting incl. leverage)."""
    r = np.log(prices).diff()
    signal = 0.5 * (r["guitars"].shift(1) + r["slingshots"].shift(1))
    sigma_sugar = r["sugar"].rolling(vol_window).std() * np.sqrt(DAYS_PER_YEAR)
    scale = (target_vol_ann / sigma_sugar).shift(1)  # no cap
    w_sugar = (signal > 0).astype(float) * scale
    port_r = w_sugar * r["sugar"]
    cash_w = 1.0 - w_sugar
    # cash_w can be negative when scale > 1 -- borrow at RF_DAILY
    return (port_r + cash_w * RF_DAILY).dropna()


def report(name, r):
    s = sharpe(r)
    lo, hi = lo_ci(r)
    print(f"  {name:38s}  Sharpe = {s:+.3f}  CI [{lo:+.3f}, {hi:+.3f}]  "
          f"MaxDD = {max_dd(r):.1%}  Terminal = ${terminal(r):,.2f}")


if __name__ == "__main__":
    prices = build_price_panel()
    print("\nLead-lag sugar variants — Sharpe, Lo-CI, MaxDD, Terminal")
    print("=" * 110)
    rA = variant_A_binary_eqw(prices)
    rB = variant_B_full(prices)
    rD = variant_D_eqw_vol(prices)
    rDp = variant_Dp_full_vol(prices)
    rDpu = variant_Dp_uncapped(prices)
    report("A) 1/7 binary  (in report)", rA)
    report("B) 100% binary (in report)", rB)
    report("D) 1/7 + vol-scaled (cap=1)", rD)
    report("D') 100% + vol-scaled (cap=1)", rDp)
    report("D'') 100% + vol-scaled (uncapped)", rDpu)
    print("=" * 110)

    # Check: average sugar exposure for each variant
    r = np.log(prices).diff()
    signal = 0.5 * (r["guitars"].shift(1) + r["slingshots"].shift(1))
    sigma_sugar = r["sugar"].rolling(20).std() * np.sqrt(DAYS_PER_YEAR)
    scale_capped = (0.30 / sigma_sugar).clip(upper=1.0).shift(1)
    scale_uncap = (0.30 / sigma_sugar).shift(1)
    w_A = (signal > 0).astype(float) * (1/7)
    w_B = (signal > 0).astype(float)
    w_D = (signal > 0).astype(float) * (1/7) * scale_capped
    w_Dp = (signal > 0).astype(float) * scale_capped
    w_Dpu = (signal > 0).astype(float) * scale_uncap
    print("\nAverage sugar exposure (only days when signal > 0, fraction of NAV):")
    pos = signal > 0
    print(f"  A:    {w_A[pos].mean():.3f}")
    print(f"  B:    {w_B[pos].mean():.3f}")
    print(f"  D:    {w_D[pos].mean():.3f}")
    print(f"  D':   {w_Dp[pos].mean():.3f}")
    print(f"  D'':  {w_Dpu[pos].mean():.3f}")
