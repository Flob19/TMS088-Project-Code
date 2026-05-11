"""Compute the four lead-lag sugar position-sizing variants and report
all 5 walk-forward sub-period Sharpe ratios for each.

Variants:
  V1) 1/7 binary long-only       (cash on neg signal, others 1/7)
  V2) 100% all-in binary         (full to sugar on pos, cash on neg)
  V3) 100% long/short            (full long on pos, full short on neg)
  V4) 100% pro-vol 20-day L/S    (long/short, sized by sigma/target)
"""
import numpy as np
import pandas as pd
from task4_strategies import (build_price_panel, ASSETS, RF_DAILY,
                              RF_ANNUAL, DAYS_PER_YEAR, metrics)

DPY = DAYS_PER_YEAR

def wf_5period_sharpes(r):
    """Per-period annualised Sharpe using the same metrics() convention."""
    r = pd.Series(r).dropna().reset_index(drop=True)
    n = len(r)
    chunk = n // 5
    out = []
    for k in range(5):
        lo, hi = k*chunk, (k+1)*chunk if k < 4 else n
        sub = r.iloc[lo:hi]
        out.append(float(metrics(sub)["sharpe"]))
    return out

def V1_binary_eqw(prices):
    """1/7 long-only binary."""
    r = np.log(prices).diff()
    sig = 0.5 * (r["guitars"].shift(1) + r["slingshots"].shift(1))
    w = pd.DataFrame(0.0, index=prices.index, columns=ASSETS)
    for a in ASSETS:
        if a != "sugar":
            w[a] = 1.0 / 7
    w["sugar"] = (sig > 0).astype(float) * (1.0 / 7)
    port_r = (w * r).sum(axis=1)
    cash_w = 1.0 - w.sum(axis=1)
    return (port_r + cash_w * RF_DAILY).dropna()

def V2_full_long(prices):
    """100% long-only binary."""
    r = np.log(prices).diff()
    sig = 0.5 * (r["guitars"].shift(1) + r["slingshots"].shift(1))
    w_sugar = (sig > 0).astype(float)
    port_r = w_sugar * r["sugar"]
    cash_w = 1.0 - w_sugar
    return (port_r + cash_w * RF_DAILY).dropna()

def V3_full_longshort(prices):
    """100% long/short binary -- full long on pos signal, full short on neg."""
    r = np.log(prices).diff()
    sig = 0.5 * (r["guitars"].shift(1) + r["slingshots"].shift(1))
    # +1 if signal > 0, -1 if signal < 0, 0 if signal == 0 (degenerate, treated as cash)
    w_sugar = np.where(sig > 0, 1.0, np.where(sig < 0, -1.0, 0.0))
    w_sugar = pd.Series(w_sugar, index=sig.index)
    port_r = w_sugar * r["sugar"]
    # When fully long or short, no cash. When sig == 0, full cash.
    cash_w = 1.0 - w_sugar.abs()
    return (port_r + cash_w * RF_DAILY).dropna()

def V4_pro_vol_LS(prices, ref_window=20, target_vol_ann=0.30, lev_cap=2.0):
    """100% pro-vol long/short, 20-day window.

    Pro-vol sizing: scale exposure by recent_sigma / long_run_sigma so the
    position is LARGER when vol is in a high-vol regime (volatility clustering
    -> bigger expected return magnitude per the same directional signal).

    w_sugar = sign(signal) * (sigma_recent / target), capped at +/- lev_cap.
    On signal == 0 -> cash.
    """
    r = np.log(prices).diff()
    sig = 0.5 * (r["guitars"].shift(1) + r["slingshots"].shift(1))
    sigma_rec = r["sugar"].rolling(ref_window).std() * np.sqrt(DPY)
    scale = (sigma_rec / target_vol_ann).clip(upper=lev_cap).shift(1)
    sign = np.where(sig > 0, 1.0, np.where(sig < 0, -1.0, 0.0))
    sign = pd.Series(sign, index=sig.index)
    w_sugar = sign * scale
    port_r = w_sugar * r["sugar"]
    # Cash leg: 1 - |w_sugar|; can be negative (borrow at RF)
    cash_w = 1.0 - w_sugar.abs()
    return (port_r + cash_w * RF_DAILY).dropna()


def report(name, r):
    m = metrics(r)
    sp = wf_5period_sharpes(r)
    pos = sum(1 for x in sp if x > 0)
    print(f"\n{name}")
    print(f"  Ann ret:   {m['ann_return']*100:+.2f}%")
    print(f"  Ann vol:   {m['ann_vol']*100:.2f}%")
    print(f"  Sharpe:    {m['sharpe']:+.3f}  CI [{m['sharpe_lo']:+.3f}, {m['sharpe_hi']:+.3f}]")
    print(f"  Max DD:    {m['max_drawdown']*100:.1f}%")
    print(f"  Terminal:  ${m['terminal_equity']:,.2f}")
    print(f"  WF (pos):  {pos}/5")
    print(f"  WF (5):    {[round(x,2) for x in sp]}")

if __name__ == "__main__":
    prices = build_price_panel()
    rs = {
        "V1) 1/7 long-only binary":         V1_binary_eqw(prices),
        "V2) 100% all-in long-only":        V2_full_long(prices),
        "V3) 100% long/short binary":       V3_full_longshort(prices),
        "V4) 100% pro-vol 20d long/short":  V4_pro_vol_LS(prices),
    }
    for k, r in rs.items():
        report(k, r)

    # Also dump all baseline strategies' 5-period Sharpes for the main table
    from task4_strategies import (strat_buy_and_hold, strat_ma_crossover,
                                   strat_inverse_vol, strat_momentum,
                                   strat_channel_breakout, strat_buy_dips,
                                   strat_short_stocks, strat_garch_timing)
    print("\n" + "=" * 70)
    print("BASELINE STRATEGIES — 5 sub-period Sharpes")
    print("=" * 70)
    baselines = {
        "Buy & Hold (EW)":          strat_buy_and_hold(prices),
        "MA crossover 20/100":      strat_ma_crossover(prices),
        "Inverse-vol risk parity":  strat_inverse_vol(prices),
        "Momentum top-3":           strat_momentum(prices),
        "Channel breakout 55/20":   strat_channel_breakout(prices),
        "Buy dips":                 strat_buy_dips(prices),
        "Short stocks":             strat_short_stocks(prices),
        "GARCH vol timing":         strat_garch_timing(prices),
    }
    for k, r in baselines.items():
        sp = wf_5period_sharpes(r)
        pos = sum(1 for x in sp if x > 0)
        print(f"  {k:30s}  {pos}/5  per-period: {[round(x,2) for x in sp]}")
