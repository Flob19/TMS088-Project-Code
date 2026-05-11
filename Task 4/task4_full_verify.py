"""Verify both lead-lag variants with the SAME Lo CI convention as main table.

Main table uses N_yr in the Lo formula: SE_ann = sqrt((1 + S^2/2)/N_yr).
"""
import numpy as np
import pandas as pd
from task4_strategies import build_price_panel, ASSETS, RF_DAILY, DAYS_PER_YEAR

DPY = DAYS_PER_YEAR

def sharpe(r):
    r = pd.Series(r).dropna()
    excess = r - RF_DAILY
    return np.sqrt(DPY) * excess.mean() / excess.std()

def lo_ci_yr(r, alpha=0.05):
    """Lo (2002) CI on ANN Sharpe using N_years (matches main table)."""
    r = pd.Series(r).dropna()
    n_yr = len(r) / DPY
    s = sharpe(r)
    se = np.sqrt((1 + 0.5 * s**2) / n_yr)
    return s - 1.96*se, s + 1.96*se

def max_dd(r):
    r = pd.Series(r).dropna()
    eq = (1 + r).cumprod()
    return float((eq / eq.cummax() - 1).min())

def ann_ret(r):
    r = pd.Series(r).dropna()
    return DPY * r.mean()

def ann_vol(r):
    r = pd.Series(r).dropna()
    return np.sqrt(DPY) * r.std()

def terminal(r):
    r = pd.Series(r).dropna()
    return float((1 + r).cumprod().iloc[-1])

def wf_5period(r):
    r = pd.Series(r).dropna().reset_index(drop=True)
    n = len(r)
    chunk = n // 5
    sharpes = []
    for k in range(5):
        lo, hi = k*chunk, (k+1)*chunk if k < 4 else n
        sub = r.iloc[lo:hi]
        excess = sub - RF_DAILY
        s = np.sqrt(DPY) * excess.mean() / excess.std()
        sharpes.append(float(s))
    pos = sum(1 for s in sharpes if s > 0)
    return pos, min(sharpes), max(sharpes), sharpes

def variant_A(prices):
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

def variant_B(prices):
    r = np.log(prices).diff()
    signal = 0.5 * (r["guitars"].shift(1) + r["slingshots"].shift(1))
    w_sugar = (signal > 0).astype(float)
    port_r = w_sugar * r["sugar"]
    cash_w = 1.0 - w_sugar
    return (port_r + cash_w * RF_DAILY).dropna()

if __name__ == "__main__":
    prices = build_price_panel()
    for name, fn in [("A) 1/7 long-only", variant_A),
                     ("B) 100% all-in",   variant_B)]:
        r = fn(prices)
        s = sharpe(r)
        lo, hi = lo_ci_yr(r)
        ar = ann_ret(r)
        av = ann_vol(r)
        dd = max_dd(r)
        term = terminal(r)
        pos, mn, mx, srs = wf_5period(r)
        print(f"\n{name}")
        print(f"  Ann ret:   {ar*100:+.2f}%")
        print(f"  Ann vol:   {av*100:.2f}%")
        print(f"  Sharpe:    {s:+.3f}  CI [{lo:+.3f}, {hi:+.3f}]")
        print(f"  Max DD:    {dd*100:.1f}%")
        print(f"  Terminal:  ${term:,.2f}")
        print(f"  WF (5):    {pos}/5  range [{mn:+.2f}, {mx:+.2f}]")
        print(f"  WF (per):  {[round(s,2) for s in srs]}")
