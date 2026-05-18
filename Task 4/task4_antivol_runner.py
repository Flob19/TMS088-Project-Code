"""Re-run Task 4 V1-V4 backtest + V4 sensitivity-grid with anti-vol V4.

Two runs:
  A) V1-V4 with block-bootstrap Sharpe CIs (block=20, B=2000, seed=20260414)
  B) V4 sensitivity grid (anti-vol): target_vol_ann, lev_cap, ref_window

Outputs:
  /sessions/charming-kind-dirac/mnt/outputs/task4_antivol_leadlag_variants.csv
  /sessions/charming-kind-dirac/mnt/outputs/task4_antivol_sensitivity.csv
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd

from task4_strategies import (
    build_price_panel, metrics, RF_ANNUAL, RF_DAILY, DAYS_PER_YEAR, ASSETS,
)
from task4_leadlag_variants import (
    V1_binary_eqw, V2_full_long, V3_full_longshort, V4_anti_vol_LS,
    wf_5period_sharpes,
)


def block_bootstrap_sharpe_ci(r, block_len=20, B=2000, seed=20260414, alpha=0.05):
    """Block-bootstrap 95% CI on annualised Sharpe."""
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


def summarise(r):
    m = metrics(r)
    wf = wf_5period_sharpes(r)
    lo, hi = block_bootstrap_sharpe_ci(r)
    return dict(
        n=m["n"],
        ann_return_pct=m["ann_return"] * 100,
        ann_vol_pct=m["ann_vol"] * 100,
        sharpe=m["sharpe"],
        sharpe_ci_lo=lo,
        sharpe_ci_hi=hi,
        max_dd_pct=m["max_drawdown"] * 100,
        terminal=m["terminal_equity"],
        wf=wf,
        wf_pos=sum(1 for x in wf if np.isfinite(x) and x > 0),
    )


def main():
    prices = build_price_panel()
    print(f"Price panel: {len(prices)} days x {len(ASSETS)} assets\n")

    # ----------------------- Run A: V1-V4 -------------------------------
    variants = {
        "V1": ("1/7 binary long-only",       V1_binary_eqw(prices)),
        "V2": ("100% all-in long-only",      V2_full_long(prices)),
        "V3": ("100% long/short binary",     V3_full_longshort(prices)),
        "V4": ("100% anti-vol 20d L/S",      V4_anti_vol_LS(prices,
                                              ref_window=20,
                                              target_vol_ann=0.30,
                                              lev_cap=2.0)),
    }

    rows_A = []
    print("=" * 78)
    print("Run A: V1-V4 (tab:leadlag_variants)  [block bootstrap CI, B=2000, "
          "block=20, seed=20260414]")
    print("=" * 78)
    for k, (label, r) in variants.items():
        s = summarise(r)
        rows_A.append(dict(
            variant=k, label=label, n_days=s["n"],
            ann_return_pct=s["ann_return_pct"],
            ann_vol_pct=s["ann_vol_pct"],
            sharpe=s["sharpe"],
            sharpe_ci_lo=s["sharpe_ci_lo"],
            sharpe_ci_hi=s["sharpe_ci_hi"],
            max_dd_pct=s["max_dd_pct"],
            terminal_equity=s["terminal"],
            wf_pos_count=s["wf_pos"],
            wf_p1=s["wf"][0], wf_p2=s["wf"][1], wf_p3=s["wf"][2],
            wf_p4=s["wf"][3], wf_p5=s["wf"][4],
        ))
        print(f"\n{k}) {label}   n={s['n']}")
        print(f"   AnnRet   {s['ann_return_pct']:+10.2f}%")
        print(f"   AnnVol   {s['ann_vol_pct']:10.2f}%")
        print(f"   Sharpe   {s['sharpe']:+10.3f}  CI95 [{s['sharpe_ci_lo']:+.3f}, "
              f"{s['sharpe_ci_hi']:+.3f}]")
        print(f"   MaxDD    {s['max_dd_pct']:10.2f}%")
        print(f"   Terminal {s['terminal']:.4e}")
        print(f"   WF 5     {[round(x,2) for x in s['wf']]}   pos {s['wf_pos']}/5")

    dfA = pd.DataFrame(rows_A)
    outA = HERE / "task4_antivol_leadlag_variants.csv"
    dfA.to_csv(outA, index=False)
    print(f"\nSaved -> {outA}")

    # LaTeX-ready block (mimicking tab:leadlag_variants row format)
    print("\n" + "=" * 78)
    print("LaTeX rows (variant & AnnRet & AnnVol & Sharpe [CI] & MaxDD & Terminal "
          "& WF+/5)")
    print("=" * 78)
    for row in rows_A:
        terminal = row["terminal_equity"]
        # scientific notation for very large terminals
        if abs(terminal) >= 1e6:
            term_str = f"{terminal:.2e}"
            # convert 7.17e+11 -> 7.17\\times10^{11}
            mant, exp = term_str.split("e")
            term_str = f"${mant}\\!\\times\\!10^{{{int(exp)}}}$"
        else:
            term_str = f"\\${terminal:,.2f}"
        print(
            f"{row['variant']} & {row['ann_return_pct']:+.2f}\\% & "
            f"{row['ann_vol_pct']:.2f}\\% & "
            f"{row['sharpe']:+.3f}\\;[{row['sharpe_ci_lo']:+.2f},\\,"
            f"{row['sharpe_ci_hi']:+.2f}] & "
            f"{row['max_dd_pct']:.1f}\\% & {term_str} & "
            f"{row['wf_pos_count']}/5 \\\\"
        )

    # ----------------------- Run B: V4 sensitivity ----------------------
    print("\n" + "=" * 78)
    print("Run B: V4 sensitivity grid (anti-vol)")
    print("=" * 78)

    rows_B = []

    def row_v4(param_name, val, **kw):
        r = V4_anti_vol_LS(prices, **kw)
        m = metrics(r)
        wf = wf_5period_sharpes(r)
        pos = sum(1 for x in wf if np.isfinite(x) and x > 0)
        rec = dict(
            strategy="V4 lead-lag (anti-vol)",
            parameter=param_name,
            value=val,
            sharpe=m["sharpe"],
            max_dd_pct=100 * m["max_drawdown"],
            wf_positive_count=pos,
        )
        rows_B.append(rec)
        return rec

    print("\nV4 target_vol_ann (ref_window=20, lev_cap=2.0)")
    for tv in [0.20, 0.30, 0.40]:
        rec = row_v4("target_vol_ann", tv,
                     ref_window=20, target_vol_ann=tv, lev_cap=2.0)
        bl = " [baseline]" if tv == 0.30 else ""
        print(f"  {tv:.2f}   Sharpe {rec['sharpe']:+.2f}  MaxDD "
              f"{rec['max_dd_pct']:+.1f}%  wf+ {rec['wf_positive_count']}/5{bl}")

    print("\nV4 lev_cap (ref_window=20, target_vol_ann=0.30)")
    for lc in [1.5, 2.0, 3.0]:
        rec = row_v4("lev_cap", lc,
                     ref_window=20, target_vol_ann=0.30, lev_cap=lc)
        bl = " [baseline]" if lc == 2.0 else ""
        print(f"  {lc:.1f}    Sharpe {rec['sharpe']:+.2f}  MaxDD "
              f"{rec['max_dd_pct']:+.1f}%  wf+ {rec['wf_positive_count']}/5{bl}")

    print("\nV4 ref_window (target_vol_ann=0.30, lev_cap=2.0)")
    for rw in [10, 20, 40]:
        rec = row_v4("ref_window", rw,
                     ref_window=rw, target_vol_ann=0.30, lev_cap=2.0)
        bl = " [baseline]" if rw == 20 else ""
        print(f"  {rw:3d}    Sharpe {rec['sharpe']:+.2f}  MaxDD "
              f"{rec['max_dd_pct']:+.1f}%  wf+ {rec['wf_positive_count']}/5{bl}")

    dfB = pd.DataFrame(rows_B,
                       columns=["strategy", "parameter", "value", "sharpe",
                                "max_dd_pct", "wf_positive_count"])
    outB = HERE / "task4_antivol_sensitivity.csv"
    dfB.to_csv(outB, index=False)
    print(f"\nSaved -> {outB}")


if __name__ == "__main__":
    main()
