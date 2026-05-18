"""Run V4 anti-vol L/S strategy with sigma_target = 0.24 (vs baseline 0.30).

Keeps ref_window=20, lev_cap=2.0.
Reports same metrics as the baseline so they can be compared side-by-side.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd

from task4_strategies import (
    build_price_panel, metrics, RF_ANNUAL, DAYS_PER_YEAR, ASSETS,
)
from task4_leadlag_variants import V4_anti_vol_LS, wf_5period_sharpes


def block_bootstrap_sharpe_ci(r, block_len=20, B=2000, seed=20260414, alpha=0.05):
    r = np.asarray(pd.Series(r).dropna().values, dtype=float)
    n = len(r)
    if n < block_len + 1:
        return (np.nan, np.nan)
    n_blocks = int(np.ceil(n / block_len))
    rng = np.random.default_rng(seed)
    sharpes = np.empty(B)
    for b in range(B):
        starts = rng.integers(0, n - block_len + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block_len)[None, :]).ravel()[:n]
        sample = r[idx]
        mu = sample.mean() * DAYS_PER_YEAR
        sd = sample.std(ddof=1) * np.sqrt(DAYS_PER_YEAR)
        sharpes[b] = (mu - RF_ANNUAL) / sd if sd > 0 else np.nan
    lo = np.nanpercentile(sharpes, 100 * alpha / 2)
    hi = np.nanpercentile(sharpes, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def report(label, r):
    m = metrics(r)
    wf = wf_5period_sharpes(r)
    pos = sum(1 for x in wf if np.isfinite(x) and x > 0)
    lo, hi = block_bootstrap_sharpe_ci(r)
    print(f"\n{label}")
    print(f"  n_days   {m['n']}")
    print(f"  AnnRet   {m['ann_return']*100:+.2f}%")
    print(f"  AnnVol   {m['ann_vol']*100:.2f}%")
    print(f"  Sharpe   {m['sharpe']:+.3f}  CI95 [{lo:+.3f}, {hi:+.3f}]")
    print(f"  MaxDD    {m['max_drawdown']*100:.2f}%")
    print(f"  Terminal {m['terminal_equity']:.4e}")
    print(f"  WF 5     {[round(x,2) for x in wf]}   pos {pos}/5")
    return dict(
        label=label,
        n=m["n"],
        ann_return_pct=m["ann_return"]*100,
        ann_vol_pct=m["ann_vol"]*100,
        sharpe=m["sharpe"],
        sharpe_ci_lo=lo,
        sharpe_ci_hi=hi,
        max_dd_pct=m["max_drawdown"]*100,
        terminal=m["terminal_equity"],
        wf=wf,
        wf_pos=pos,
    )


def main():
    prices = build_price_panel()
    print(f"Price panel: {len(prices)} days x {len(ASSETS)} assets")

    r024 = V4_anti_vol_LS(prices, ref_window=20, target_vol_ann=0.24, lev_cap=2.0)
    s024 = report("V4 anti-vol L/S  sigma_target=0.24, ref_window=20, lev_cap=2.0",
                  r024)

    # Also re-compute the 0.30 baseline so we can compare with identical pipeline
    r030 = V4_anti_vol_LS(prices, ref_window=20, target_vol_ann=0.30, lev_cap=2.0)
    s030 = report("V4 anti-vol L/S  sigma_target=0.30 [baseline]",
                  r030)

    print("\n" + "="*78)
    print("Side-by-side: sigma_target 0.24 vs 0.30")
    print("="*78)
    rows = [
        ("AnnRet %",     f"{s024['ann_return_pct']:+.2f}",  f"{s030['ann_return_pct']:+.2f}"),
        ("AnnVol %",     f"{s024['ann_vol_pct']:.2f}",      f"{s030['ann_vol_pct']:.2f}"),
        ("Sharpe (rf=3%)", f"{s024['sharpe']:+.3f}",        f"{s030['sharpe']:+.3f}"),
        ("Sharpe CI lo", f"{s024['sharpe_ci_lo']:+.3f}",    f"{s030['sharpe_ci_lo']:+.3f}"),
        ("Sharpe CI hi", f"{s024['sharpe_ci_hi']:+.3f}",    f"{s030['sharpe_ci_hi']:+.3f}"),
        ("MaxDD %",      f"{s024['max_dd_pct']:.2f}",       f"{s030['max_dd_pct']:.2f}"),
        ("Terminal",     f"{s024['terminal']:.3e}",         f"{s030['terminal']:.3e}"),
        ("WF pos / 5",   f"{s024['wf_pos']}",                f"{s030['wf_pos']}"),
        ("WF Sharpes",   str([round(x,2) for x in s024['wf']]),
                         str([round(x,2) for x in s030['wf']])),
    ]
    print(f"{'Metric':<18}{'sigma=0.24':>22}{'sigma=0.30':>22}")
    for name, a, b in rows:
        print(f"{name:<18}{a:>22}{b:>22}")


if __name__ == "__main__":
    main()
