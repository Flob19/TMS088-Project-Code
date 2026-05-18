"""Parameter sensitivity grid for Task 4 strategies.

Runs 7 single-parameter grids (MA crossover, inverse-vol, momentum lookback,
momentum top_k, V4 target_vol_ann, V4 lev_cap, V4 ref_window) and reports
Sharpe, max drawdown, and the count of walk-forward 5-period Sharpes > 0.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd

from task4_strategies import (
    build_price_panel, strat_ma_crossover, strat_inverse_vol, strat_momentum,
    metrics, RF_ANNUAL,
)
from task4_leadlag_variants import V4_pro_vol_LS, wf_5period_sharpes


def run_one(returns):
    m = metrics(returns)
    wf = wf_5period_sharpes(returns)
    pos = sum(1 for x in wf if (x is not None and np.isfinite(x) and x > 0))
    return m["sharpe"], 100 * m["max_drawdown"], pos


def fmt_row(label, sharpe, dd, wf_pos, baseline=False):
    tag = "   [baseline]" if baseline else ""
    return f"  {label:14s} Sharpe {sharpe:+.2f}  MaxDD {dd:+.1f}%  wf+ {wf_pos}/5{tag}"


def main():
    prices = build_price_panel()
    print(f"Price panel: {len(prices)} days x 7 assets")
    print()

    rows = []  # rows for csv

    # ---- 1. MA crossover (short, long) ----
    print("MA crossover (short, long)")
    ma_grid = [(10, 50), (20, 100), (50, 200)]
    ma_baseline = (20, 100)
    ma_sharpes = []
    for s, l in ma_grid:
        r = strat_ma_crossover(prices, short=s, long=l)
        sh, dd, pos = run_one(r)
        ma_sharpes.append(sh)
        rows.append(dict(strategy="MA crossover", parameter="(short,long)",
                         value=f"({s},{l})", sharpe=sh, max_dd_pct=dd,
                         wf_positive_count=pos))
        print(fmt_row(f"({s}, {l}):", sh, dd, pos, baseline=((s, l) == ma_baseline)))
    print()

    # ---- 2. Inverse-vol vol_window ----
    print("Inverse-vol vol_window")
    iv_grid = [30, 60, 120]
    iv_baseline = 60
    iv_sharpes = []
    for vw in iv_grid:
        r = strat_inverse_vol(prices, vol_window=vw)
        sh, dd, pos = run_one(r)
        iv_sharpes.append(sh)
        rows.append(dict(strategy="Inverse-vol", parameter="vol_window",
                         value=vw, sharpe=sh, max_dd_pct=dd,
                         wf_positive_count=pos))
        print(fmt_row(f"{vw}:", sh, dd, pos, baseline=(vw == iv_baseline)))
    print()

    # ---- 3. Momentum lookback ----
    print("Momentum lookback (top_k=3, rebal_every=21)")
    mom_lb_grid = [40, 60, 90]
    mom_lb_baseline = 60
    mom_lb_sharpes = []
    for lb in mom_lb_grid:
        r = strat_momentum(prices, lookback=lb, top_k=3, rebal_every=21)
        sh, dd, pos = run_one(r)
        mom_lb_sharpes.append(sh)
        rows.append(dict(strategy="Momentum", parameter="lookback",
                         value=lb, sharpe=sh, max_dd_pct=dd,
                         wf_positive_count=pos))
        print(fmt_row(f"{lb}:", sh, dd, pos, baseline=(lb == mom_lb_baseline)))
    print()

    # ---- 4. Momentum top_k ----
    print("Momentum top_k (lookback=60, rebal_every=21)")
    mom_k_grid = [2, 3, 4]
    mom_k_baseline = 3
    mom_k_sharpes = []
    for k in mom_k_grid:
        r = strat_momentum(prices, lookback=60, top_k=k, rebal_every=21)
        sh, dd, pos = run_one(r)
        mom_k_sharpes.append(sh)
        rows.append(dict(strategy="Momentum", parameter="top_k",
                         value=k, sharpe=sh, max_dd_pct=dd,
                         wf_positive_count=pos))
        print(fmt_row(f"{k}:", sh, dd, pos, baseline=(k == mom_k_baseline)))
    print()

    # ---- 5. V4 target_vol_ann ----
    print("V4 lead-lag sugar target_vol_ann (ref_window=20, lev_cap=2.0)")
    v4_tv_grid = [0.20, 0.30, 0.40]
    v4_tv_baseline = 0.30
    v4_tv_sharpes = []
    for tv in v4_tv_grid:
        r = V4_pro_vol_LS(prices, ref_window=20, target_vol_ann=tv, lev_cap=2.0)
        sh, dd, pos = run_one(r)
        v4_tv_sharpes.append(sh)
        rows.append(dict(strategy="V4 lead-lag", parameter="target_vol_ann",
                         value=tv, sharpe=sh, max_dd_pct=dd,
                         wf_positive_count=pos))
        print(fmt_row(f"{tv:.2f}:", sh, dd, pos, baseline=(tv == v4_tv_baseline)))
    print()

    # ---- 6. V4 lev_cap ----
    print("V4 lev_cap (ref_window=20, target_vol_ann=0.30)")
    v4_lc_grid = [1.5, 2.0, 3.0]
    v4_lc_baseline = 2.0
    v4_lc_sharpes = []
    for lc in v4_lc_grid:
        r = V4_pro_vol_LS(prices, ref_window=20, target_vol_ann=0.30, lev_cap=lc)
        sh, dd, pos = run_one(r)
        v4_lc_sharpes.append(sh)
        rows.append(dict(strategy="V4 lead-lag", parameter="lev_cap",
                         value=lc, sharpe=sh, max_dd_pct=dd,
                         wf_positive_count=pos))
        print(fmt_row(f"{lc:.1f}:", sh, dd, pos, baseline=(lc == v4_lc_baseline)))
    print()

    # ---- 7. V4 ref_window ----
    print("V4 ref_window (target_vol_ann=0.30, lev_cap=2.0)")
    v4_rw_grid = [10, 20, 40]
    v4_rw_baseline = 20
    v4_rw_sharpes = []
    for rw in v4_rw_grid:
        r = V4_pro_vol_LS(prices, ref_window=rw, target_vol_ann=0.30, lev_cap=2.0)
        sh, dd, pos = run_one(r)
        v4_rw_sharpes.append(sh)
        rows.append(dict(strategy="V4 lead-lag", parameter="ref_window",
                         value=rw, sharpe=sh, max_dd_pct=dd,
                         wf_positive_count=pos))
        print(fmt_row(f"{rw}:", sh, dd, pos, baseline=(rw == v4_rw_baseline)))
    print()

    # ---- Robustness verdict ----
    print("=" * 70)
    print("ROBUSTNESS VERDICT")
    print("=" * 70)

    def verdict(label, grid_vals, grid_sharpes, baseline_value):
        smin, smax = min(grid_sharpes), max(grid_sharpes)
        sorted_s = sorted(grid_sharpes)
        baseline_idx = grid_vals.index(baseline_value)
        baseline_s = grid_sharpes[baseline_idx]
        # midpoint check: is baseline the middle of the sorted Sharpe sequence?
        median_s = sorted_s[len(sorted_s) // 2]
        mid_yes = abs(baseline_s - median_s) < 1e-9
        # edge check: is baseline_s the min or max?
        is_edge = (baseline_s == smin) or (baseline_s == smax)
        # stable plateau: range small (max-min < 0.10) or baseline median
        range_w = smax - smin
        if range_w < 0.10:
            plateau_tag = "stable plateau (range < 0.10)"
        elif is_edge:
            plateau_tag = "near an edge"
        else:
            plateau_tag = "stable plateau (baseline interior)"
        mid_str = "yes" if mid_yes else "no"
        print(f"{label}")
        print(f"  Sharpe range across grid: [{smin:+.2f}, {smax:+.2f}]; "
              f"baseline at midpoint? {mid_str}")
        print(f"  Baseline sits in {plateau_tag} "
              f"(range={range_w:.2f}, baseline_S={baseline_s:+.2f})")

    verdict("MA crossover", ma_grid, ma_sharpes, ma_baseline)
    verdict("Inverse-vol vol_window", iv_grid, iv_sharpes, iv_baseline)
    verdict("Momentum lookback", mom_lb_grid, mom_lb_sharpes, mom_lb_baseline)
    verdict("Momentum top_k", mom_k_grid, mom_k_sharpes, mom_k_baseline)
    verdict("V4 target_vol_ann", v4_tv_grid, v4_tv_sharpes, v4_tv_baseline)
    verdict("V4 lev_cap", v4_lc_grid, v4_lc_sharpes, v4_lc_baseline)
    verdict("V4 ref_window", v4_rw_grid, v4_rw_sharpes, v4_rw_baseline)

    df = pd.DataFrame(rows, columns=["strategy", "parameter", "value",
                                     "sharpe", "max_dd_pct",
                                     "wf_positive_count"])
    out_csv = HERE / "task4_sens_results.csv"
    df.to_csv(out_csv, index=False)
    print()
    print(f"Saved -> {out_csv}")


if __name__ == "__main__":
    main()
