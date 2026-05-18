"""Task 4 full re-run after Task 2 sugar gap-fill patch.

Runs:
  A. Strategy performance table for the 5 production strategies.
  B. Lead-lag variants V1-V4.
  C. Sensitivity grid (7 single-parameter grids).

Bootstrap CI on Sharpe: block bootstrap, 20-day blocks, 2000 resamples,
seed 20260414. CI = [2.5, 97.5] percentiles.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Import production code in-place (uses patched task2_interpolation.py)
TASK4 = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK4))

from task4_strategies import (
    build_price_panel, strat_buy_and_hold, strat_ma_crossover,
    strat_inverse_vol, strat_momentum, strat_leadlag_sugar,
    metrics, RF_ANNUAL, DAYS_PER_YEAR,
)
from task4_leadlag_variants import (
    V1_binary_eqw, V2_full_long, V3_full_longshort, V4_pro_vol_LS,
    wf_5period_sharpes,
)

DPY = DAYS_PER_YEAR
OUT = Path(__file__).resolve().parent


def block_bootstrap_sharpe_ci(returns, n_resamples=2000, block=20,
                              seed=20260414, rf_annual=RF_ANNUAL):
    r = np.asarray(pd.Series(returns).dropna())
    N = len(r)
    rng = np.random.default_rng(seed)
    n_blocks = N // block
    if n_blocks < 1:
        return np.nan, np.nan
    sharpes = np.empty(n_resamples)
    for k in range(n_resamples):
        starts = rng.integers(0, N - block + 1, size=n_blocks)
        sample = np.concatenate([r[s:s + block] for s in starts])
        mu = sample.mean() * DPY
        sd = sample.std(ddof=1) * np.sqrt(DPY)
        sharpes[k] = (mu - rf_annual) / sd if sd > 0 else np.nan
    return float(np.nanpercentile(sharpes, 2.5)), float(np.nanpercentile(sharpes, 97.5))


def run_strategy(name, returns):
    m = metrics(returns)
    lo, hi = block_bootstrap_sharpe_ci(returns)
    wf = wf_5period_sharpes(returns)
    pos = sum(1 for x in wf if (x is not None and np.isfinite(x) and x > 0))
    return dict(
        strategy=name,
        n_days=m["n"],
        ann_return_pct=100 * m["ann_return"],
        ann_vol_pct=100 * m["ann_vol"],
        sharpe=m["sharpe"],
        boot_lo=lo,
        boot_hi=hi,
        max_dd_pct=100 * m["max_drawdown"],
        terminal_equity=m["terminal_equity"],
        sub1=wf[0], sub2=wf[1], sub3=wf[2], sub4=wf[3], sub5=wf[4],
        wf_positive=pos,
    )


def main():
    prices = build_price_panel()
    print(f"Price panel: {len(prices)} days x 7 assets")
    print()

    # =====================================================================
    # A. Production strategies
    # =====================================================================
    print("=" * 70)
    print("RUN A: production strategies (tab:strat_perf)")
    print("=" * 70)
    strat_a = [
        ("Buy & Hold (EW)",          strat_buy_and_hold(prices)),
        ("MA crossover 20/100",      strat_ma_crossover(prices)),
        ("Inverse-vol risk parity",  strat_inverse_vol(prices)),
        ("Momentum top-3",           strat_momentum(prices)),
        ("Lead-lag sugar V1",        strat_leadlag_sugar(prices)),
    ]
    rows_a = [run_strategy(n, r) for n, r in strat_a]
    df_a = pd.DataFrame(rows_a)
    df_a.to_csv(OUT / "task4_rerun_strat_perf.csv", index=False)
    print(df_a.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()

    # =====================================================================
    # B. Lead-lag variants V1-V4
    # =====================================================================
    print("=" * 70)
    print("RUN B: lead-lag variants (tab:leadlag_variants)")
    print("=" * 70)
    strat_b = [
        ("V1) 1/7 binary",           V1_binary_eqw(prices)),
        ("V2) 100% all-in",          V2_full_long(prices)),
        ("V3) 100% long/short",      V3_full_longshort(prices)),
        ("V4) 100% anti-vol L/S",    V4_pro_vol_LS(prices, ref_window=20,
                                                  target_vol_ann=0.24,
                                                  lev_cap=2.0)),
    ]
    rows_b = [run_strategy(n, r) for n, r in strat_b]
    df_b = pd.DataFrame(rows_b)
    df_b.to_csv(OUT / "task4_rerun_leadlag_variants.csv", index=False)
    print(df_b.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()

    # =====================================================================
    # C. Sensitivity grid
    # =====================================================================
    print("=" * 70)
    print("RUN C: sensitivity grid")
    print("=" * 70)
    grid_rows = []

    def add_row(strategy, parameter, value, returns):
        m = metrics(returns)
        wf = wf_5period_sharpes(returns)
        pos = sum(1 for x in wf if (x is not None and np.isfinite(x) and x > 0))
        row = dict(
            strategy=strategy, parameter=parameter, value=value,
            sharpe=m["sharpe"], max_dd_pct=100 * m["max_drawdown"],
            wf_positive=pos,
        )
        grid_rows.append(row)
        print(f"  {strategy:14s} {parameter:14s} {str(value):>8s}  "
              f"Sharpe {m['sharpe']:+.3f}  MaxDD {100*m['max_drawdown']:+.2f}%  "
              f"wf+ {pos}/5")
        return row

    print("MA crossover (short, long):")
    for s, l in [(10, 50), (20, 100), (50, 200)]:
        add_row("MA crossover", "(short,long)", f"({s},{l})",
                strat_ma_crossover(prices, short=s, long=l))

    print("Inverse-vol vol_window:")
    for vw in [30, 60, 120]:
        add_row("Inverse-vol", "vol_window", vw,
                strat_inverse_vol(prices, vol_window=vw))

    print("Momentum lookback (top_k=3):")
    for lb in [40, 60, 90]:
        add_row("Momentum", "lookback", lb,
                strat_momentum(prices, lookback=lb, top_k=3, rebal_every=21))

    print("Momentum top_k (lookback=60):")
    for k in [2, 3, 4]:
        add_row("Momentum", "top_k", k,
                strat_momentum(prices, lookback=60, top_k=k, rebal_every=21))

    print("V4 target_vol_ann (ref_window=20, lev_cap=2.0):")
    for tv in [0.20, 0.30, 0.40]:
        add_row("V4 lead-lag", "target_vol_ann", f"{tv:.2f}",
                V4_pro_vol_LS(prices, ref_window=20, target_vol_ann=tv,
                              lev_cap=2.0))

    print("V4 lev_cap (ref_window=20, target_vol_ann=0.30):")
    for lc in [1.5, 2.0, 3.0]:
        add_row("V4 lead-lag", "lev_cap", f"{lc:.1f}",
                V4_pro_vol_LS(prices, ref_window=20, target_vol_ann=0.30,
                              lev_cap=lc))

    print("V4 ref_window (target_vol_ann=0.30, lev_cap=2.0):")
    for rw in [10, 20, 40]:
        add_row("V4 lead-lag", "ref_window", rw,
                V4_pro_vol_LS(prices, ref_window=rw, target_vol_ann=0.30,
                              lev_cap=2.0))

    df_c = pd.DataFrame(grid_rows)
    df_c.to_csv(OUT / "task4_rerun_sensitivity.csv", index=False)

    print()
    print("All CSVs written to", OUT)


if __name__ == "__main__":
    main()
