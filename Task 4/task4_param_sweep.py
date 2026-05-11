"""Parameter sensitivity for vol-scaling on lead-lag sugar.

The point: if vol-scaling is robust, Sharpe should be flat across reasonable
windows and target-vol values.  If it depends sharply on one combination, it's
spec search.
"""
import numpy as np
import pandas as pd
from task4_strategies import build_price_panel, ASSETS, RF_DAILY, DAYS_PER_YEAR

def sharpe(r):
    r = pd.Series(r).dropna()
    excess = r - RF_DAILY
    return np.sqrt(DAYS_PER_YEAR) * excess.mean() / excess.std()

def variant_eqw_vol(prices, target_vol_ann, vol_window):
    r = np.log(prices).diff()
    signal = 0.5 * (r["guitars"].shift(1) + r["slingshots"].shift(1))
    sigma_sugar = r["sugar"].rolling(vol_window).std() * np.sqrt(DAYS_PER_YEAR)
    scale = (target_vol_ann / sigma_sugar).clip(upper=1.0).shift(1)
    w = pd.DataFrame(0.0, index=prices.index, columns=ASSETS)
    for a in ASSETS:
        if a != "sugar":
            w[a] = 1.0 / 7
    w["sugar"] = (signal > 0).astype(float) * (1.0 / 7) * scale
    port_r = (w * r).sum(axis=1)
    cash_w = 1.0 - w.sum(axis=1)
    return (port_r + cash_w * RF_DAILY).dropna()

def variant_full_vol(prices, target_vol_ann, vol_window):
    r = np.log(prices).diff()
    signal = 0.5 * (r["guitars"].shift(1) + r["slingshots"].shift(1))
    sigma_sugar = r["sugar"].rolling(vol_window).std() * np.sqrt(DAYS_PER_YEAR)
    scale = (target_vol_ann / sigma_sugar).clip(upper=1.0).shift(1)
    w_sugar = (signal > 0).astype(float) * scale
    port_r = w_sugar * r["sugar"]
    cash_w = 1.0 - w_sugar
    return (port_r + cash_w * RF_DAILY).dropna()

if __name__ == "__main__":
    prices = build_price_panel()
    windows = [5, 10, 20, 40, 60]
    targets = [0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

    print("\nVariant D (1/7 + vol-scaled): Sharpe by [target_vol_ann × window]")
    print("(baseline 1/7 binary Sharpe = +0.86)")
    print(f"{'target':>10s} | " + " ".join(f"w={w:>3d}" for w in windows))
    for t in targets:
        row = [sharpe(variant_eqw_vol(prices, t, w)) for w in windows]
        print(f"{t:10.2f} | " + " ".join(f"{x:+.3f}" for x in row))

    print("\nVariant D' (100% + vol-scaled): Sharpe by [target_vol_ann × window]")
    print("(baseline 100% binary Sharpe = +3.13)")
    print(f"{'target':>10s} | " + " ".join(f"w={w:>3d}" for w in windows))
    for t in targets:
        row = [sharpe(variant_full_vol(prices, t, w)) for w in windows]
        print(f"{t:10.2f} | " + " ".join(f"{x:+.3f}" for x in row))
