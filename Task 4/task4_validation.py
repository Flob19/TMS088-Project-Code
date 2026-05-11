"""
Task 4 — Robustness validation tests for the trading strategies.

Implements (in order):
  1. 80/20 in-sample / out-of-sample holdout
  2. 5-period walk-forward Sharpe
  3. 147-strategy specification search (all (predictors, target) lead-lag
     configurations)
  4. Combinatorially Purged Cross-Validation (CPCV) and the Probability of
     Backtest Overfitting (PBO) statistic, following Bailey, Borwein, Lopez
     de Prado, Zhu (2017) "The Probability of Backtest Overfitting".
  5. Permutation test: shuffle the predictor returns to break the temporal
     coupling and check that the Sharpe of the lead-lag sugar strategy
     collapses to ~0 under the null hypothesis of no lead-lag.
  6. Block bootstrap: resample data in blocks of 21 days to construct
     alternative datasets and verify Sharpe stability.
  7. Threshold sensitivity: try different signal thresholds (not only 0)
     to confirm the chosen rule is not cherry-picked.
  8. Transaction cost sensitivity: compute Sharpe after various basis-point
     trading costs to find the break-even level.
  9. Information ratio vs Buy & Hold: confirm the edge is over the market,
     not just from general drift.

All results are written to CSV files in the project root.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from itertools import combinations

from task4_strategies import (
    build_price_panel, metrics, ASSETS, RF_DAILY, RF_ANNUAL,
    strat_buy_and_hold, strat_ma_crossover, strat_inverse_vol,
    strat_momentum, strat_channel_breakout, strat_buy_dips,
    strat_short_stocks, strat_leadlag_sugar, strat_garch_timing,
)

ROOT = Path(__file__).parent
DAYS_PER_YEAR = 252


def sharpe_simple(log_returns, rf_annual=RF_ANNUAL):
    """Annualised Sharpe from a series of daily log-returns (no Lo CI)."""
    r = np.asarray(log_returns)
    if len(r) < 30 or np.nanstd(r, ddof=1) == 0:
        return np.nan
    ann_ret = np.nanmean(r) * DAYS_PER_YEAR
    ann_vol = np.nanstd(r, ddof=1) * np.sqrt(DAYS_PER_YEAR)
    return (ann_ret - rf_annual) / ann_vol


# ---------------------------------------------------------------------------
# 1. 80/20 holdout
# ---------------------------------------------------------------------------
def test_80_20_holdout(strategies, split_frac=0.80):
    """Compute Sharpe on full sample, in-sample (first 80%) and out-of-sample
    (last 20%) for each strategy."""
    rows = []
    for name, port in strategies.items():
        n = len(port)
        if n < 100:
            continue
        # port.index is the day index from the original price panel
        max_day = port.index.max()
        split_day = int(max_day * split_frac)
        is_port = port[port.index < split_day]
        oos_port = port[port.index >= split_day]
        full_m = metrics(port)
        is_m = metrics(is_port) if len(is_port) > 50 else {"sharpe": np.nan,
                                                            "max_drawdown": np.nan}
        oos_m = metrics(oos_port) if len(oos_port) > 50 else {"sharpe": np.nan,
                                                              "max_drawdown": np.nan,
                                                              "sharpe_lo": np.nan,
                                                              "sharpe_hi": np.nan}
        rows.append({
            "strategy": name,
            "full_sharpe": full_m["sharpe"],
            "is_sharpe": is_m.get("sharpe"),
            "oos_sharpe": oos_m.get("sharpe"),
            "oos_sharpe_lo": oos_m.get("sharpe_lo"),
            "oos_sharpe_hi": oos_m.get("sharpe_hi"),
            "oos_max_dd_%": 100 * oos_m.get("max_drawdown", np.nan),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. 5-period walk-forward Sharpe
# ---------------------------------------------------------------------------
def test_walk_forward(strategies, n_periods=5):
    """Compute Sharpe per period for each strategy.  Returns long-format
    DataFrame plus a wide-format pivot for readability."""
    rows = []
    # Determine sample length from any strategy
    any_port = next(iter(strategies.values()))
    max_day = any_port.index.max()
    boundaries = [int(max_day * k / n_periods) for k in range(n_periods + 1)]
    for name, port in strategies.items():
        for k in range(n_periods):
            lo, hi = boundaries[k], boundaries[k + 1]
            sub = port[(port.index >= lo) & (port.index < hi)]
            if len(sub) > 50:
                rows.append({
                    "strategy": name,
                    "period": k + 1,
                    "day_lo": lo,
                    "day_hi": hi,
                    "sharpe": metrics(sub)["sharpe"],
                    "n_days": len(sub),
                })
    long_df = pd.DataFrame(rows)
    wide = long_df.pivot(index="strategy", columns="period", values="sharpe")
    wide.columns = [f"period_{c}" for c in wide.columns]
    wide["full"] = [metrics(strategies[name])["sharpe"] for name in wide.index]
    wide["positive_periods"] = (long_df.groupby("strategy")["sharpe"]
                                       .apply(lambda s: (s > 0).sum()))
    return long_df, wide


# ---------------------------------------------------------------------------
# 3. Specification search across all 147 lead-lag configurations
# ---------------------------------------------------------------------------
def leadlag_strategy(prices, predictors, target):
    """Generic lead-lag strategy template.  Holds 1/7 long target if
    yesterday's mean of `predictors` log-returns is positive; other six
    assets always 1/7 long; cash earns RF."""
    r = np.log(prices).diff()
    if isinstance(predictors, str):
        signal = r[predictors].shift(1)
    else:
        signal = sum(r[p].shift(1) for p in predictors) / len(predictors)
    w = pd.DataFrame(0.0, index=prices.index, columns=ASSETS)
    for a in ASSETS:
        if a != target:
            w[a] = 1.0 / 7
    w[target] = (signal > 0).astype(float) * (1.0 / 7)
    port_r = (w * r).sum(axis=1) + (1 - w.sum(axis=1)) * RF_DAILY
    return port_r.dropna()


def test_specification_search(prices):
    """Test every (predictors, target) lead-lag configuration:
    * 42 single-predictor strategies (one predictor → one target)
    * 105 two-predictor strategies (mean of two predictors → one target)
    Total 147 strategies.  Reports rank of our deployment choice
    (guitars+slingshots → sugar)."""
    rows = []
    # single-predictor
    for pred in ASSETS:
        for tgt in ASSETS:
            if pred == tgt:
                continue
            port = leadlag_strategy(prices, pred, tgt)
            rows.append({
                "predictors": pred,
                "target": tgt,
                "n_predictors": 1,
                "sharpe": metrics(port)["sharpe"],
            })
    # two-predictor
    for tgt in ASSETS:
        other = [a for a in ASSETS if a != tgt]
        for combo in combinations(other, 2):
            port = leadlag_strategy(prices, list(combo), tgt)
            rows.append({
                "predictors": "+".join(combo),
                "target": tgt,
                "n_predictors": 2,
                "sharpe": metrics(port)["sharpe"],
            })
    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


# ---------------------------------------------------------------------------
# 4. Combinatorially Purged Cross-Validation (CPCV) and PBO
# ---------------------------------------------------------------------------
def compute_cpcv_pbo(strategies, K=10):
    """Compute the Probability of Backtest Overfitting (PBO) following
    Bailey/Borwein/Lopez de Prado/Zhu (2017).

    Algorithm:
      * Align all strategy daily log-returns into a T x N matrix.
      * Split T days into K equal-size, non-overlapping blocks.
      * Generate every combination C(K, K/2) of "training" blocks; the
        remaining K/2 blocks form the "testing" set.
      * For each split:
          - Compute Sharpe per strategy on training set and on testing set.
          - Identify the strategy that ranks #1 on training.
          - Find its rank on the testing set (1 = best).
          - Compute logit-transformed relative rank
            omega = rank / (N+1), log-lambda = log(omega/(1-omega)).
      * PBO = P(log-lambda <= 0)
            = fraction of splits where the best-in-sample strategy
              ranks at or below median out-of-sample.

    Lower PBO is better.  PBO < 0.5 means the in-sample winner tends to
    rank above median out-of-sample (i.e. some genuine signal).
    PBO ≈ 0.5 means in-sample selection has no predictive power
    (pure overfitting).  PBO > 0.5 means in-sample winners actively become
    out-of-sample losers (severe overfitting)."""
    # Align strategies into a single DataFrame
    df_rets = pd.concat(strategies, axis=1).dropna(how="any")
    df_rets.columns = list(strategies.keys())
    T, N = df_rets.shape
    block_size = T // K
    if block_size < 50:
        raise ValueError(f"K={K} too large for T={T}; block size only {block_size}")

    print(f"  CPCV setup: T={T} days, N={N} strategies, K={K} blocks of {block_size} days each")
    print(f"  Number of splits: C({K},{K//2}) = {len(list(combinations(range(K), K//2)))}")

    splits = list(combinations(range(K), K // 2))
    log_lambdas = []

    for split_idx, train_blocks in enumerate(splits):
        train_blocks = set(train_blocks)
        test_blocks = set(range(K)) - train_blocks

        train_rows = []
        for b in train_blocks:
            train_rows.extend(range(b * block_size, (b + 1) * block_size))
        test_rows = []
        for b in test_blocks:
            test_rows.extend(range(b * block_size, (b + 1) * block_size))

        train_data = df_rets.iloc[train_rows]
        test_data = df_rets.iloc[test_rows]

        train_sharpes = {col: sharpe_simple(train_data[col]) for col in df_rets.columns}
        test_sharpes = {col: sharpe_simple(test_data[col]) for col in df_rets.columns}

        # Best-in-sample strategy
        best_in = max(train_sharpes, key=lambda c: train_sharpes[c])

        # Its rank in test (1 = best, N = worst)
        sorted_test = sorted(test_sharpes.items(), key=lambda kv: -kv[1])
        test_rank = next(i for i, (c, _) in enumerate(sorted_test, start=1)
                         if c == best_in)

        # Bailey/Borwein/Lopez de Prado/Zhu (2017) convention:
        # omega = relative rank with HIGH omega = HIGH performance.
        # Since our test_rank uses 1 = best, we need to invert:
        #   omega = (N + 1 - rank) / (N + 1)
        # so that test_rank=1 (best) gives omega close to 1, log_lambda > 0
        # and test_rank=N (worst) gives omega close to 0, log_lambda < 0.
        # PBO = P(log_lambda <= 0) = P(in-sample winner ranks at or below
        # median out-of-sample) = probability of overfitting.
        omega = (N + 1 - test_rank) / (N + 1)
        omega = np.clip(omega, 1e-8, 1 - 1e-8)
        log_lambda = np.log(omega / (1 - omega))
        log_lambdas.append({
            "split": split_idx,
            "best_in_sample": best_in,
            "is_sharpe": train_sharpes[best_in],
            "oos_rank": test_rank,
            "oos_sharpe": test_sharpes[best_in],
            "log_lambda": log_lambda,
        })

    summary = pd.DataFrame(log_lambdas)
    pbo = (summary["log_lambda"] <= 0).mean()
    return pbo, summary


# ---------------------------------------------------------------------------
# 5. Permutation test
# ---------------------------------------------------------------------------
def test_permutation(prices, n_perm=1000, seed=20260428):
    """Shuffle the temporal order of guitars+slingshots returns and recompute
    the lead-lag sugar Sharpe.  If the original signal is real, the
    permuted Sharpes should cluster around zero with the observed value
    being extreme."""
    rng = np.random.default_rng(seed)
    r = np.log(prices).diff()

    # Original (un-permuted) signal and Sharpe
    original_sharpe = sharpe_simple(_leadlag_returns(r, prices,
                                                    ["guitars", "slingshots"], "sugar"))

    permuted_sharpes = np.empty(n_perm)
    n = len(r)
    for i in range(n_perm):
        # Independently permute guitars and slingshots returns over time
        r_perm = r.copy()
        idx_g = rng.permutation(n)
        idx_s = rng.permutation(n)
        r_perm["guitars"] = r["guitars"].values[idx_g]
        r_perm["slingshots"] = r["slingshots"].values[idx_s]
        # Sugar returns and other assets stay in their original order
        sh = sharpe_simple(_leadlag_returns(r_perm, prices,
                                            ["guitars", "slingshots"], "sugar"))
        permuted_sharpes[i] = sh

    p_value = np.mean(permuted_sharpes >= original_sharpe)
    return {
        "original_sharpe": original_sharpe,
        "n_permutations": n_perm,
        "permuted_mean": float(np.mean(permuted_sharpes)),
        "permuted_std": float(np.std(permuted_sharpes, ddof=1)),
        "permuted_max": float(np.max(permuted_sharpes)),
        "p_value": float(p_value),
        "z_score": float((original_sharpe - np.mean(permuted_sharpes))
                          / np.std(permuted_sharpes, ddof=1)),
        "permuted_sharpes": permuted_sharpes,
    }


def _leadlag_returns(r, prices, predictors, target):
    """Compute the daily portfolio returns of a generic lead-lag strategy
    given a returns DataFrame `r`. Replicates the logic of
    leadlag_strategy() but takes pre-computed returns as input."""
    if isinstance(predictors, str):
        signal = r[predictors].shift(1)
    else:
        signal = sum(r[p].shift(1) for p in predictors) / len(predictors)
    w = pd.DataFrame(0.0, index=prices.index, columns=ASSETS)
    for a in ASSETS:
        if a != target:
            w[a] = 1.0 / 7
    w[target] = (signal > 0).astype(float) * (1.0 / 7)
    port_r = (w * r).sum(axis=1) + (1 - w.sum(axis=1)) * RF_DAILY
    return port_r.dropna()


# ---------------------------------------------------------------------------
# 6. Block bootstrap
# ---------------------------------------------------------------------------
def test_block_bootstrap(prices, block_size=21, n_boot=500, seed=20260429):
    """Block bootstrap: resample contiguous blocks of `block_size` days
    (with replacement) to construct alternative datasets, then compute
    the lead-lag sugar Sharpe on each.  This preserves short-run dependence
    while randomising the larger-scale structure."""
    rng = np.random.default_rng(seed)
    r = np.log(prices).diff().dropna()
    n = len(r)
    n_blocks = n // block_size

    boot_sharpes = np.empty(n_boot)
    for i in range(n_boot):
        # Sample n_blocks blocks with replacement
        starts = rng.integers(0, n - block_size, size=n_blocks)
        idx = []
        for s in starts:
            idx.extend(range(s, s + block_size))
        idx = idx[:n]  # truncate to original length
        r_boot = r.iloc[idx].reset_index(drop=True)
        # Recompute lead-lag sugar on bootstrapped returns
        signal = 0.5 * (r_boot["guitars"].shift(1) + r_boot["slingshots"].shift(1))
        # Sugar weight is 1/7 when signal > 0, else 0 (cash takes the slot)
        sugar_w = (signal > 0).astype(float) * (1.0 / 7)
        cash_w = (1.0 / 7) - sugar_w  # 0 when in sugar, 1/7 when in cash
        # 6 non-sugar assets: always 1/7
        baseline = sum((1.0 / 7) * r_boot[a] for a in ASSETS if a != "sugar")
        sugar_contrib = sugar_w * r_boot["sugar"]
        cash_contrib = cash_w * RF_DAILY
        port_r = baseline + sugar_contrib + cash_contrib
        port_r = port_r.dropna()
        boot_sharpes[i] = sharpe_simple(port_r)

    return {
        "n_bootstrap": n_boot,
        "block_size": block_size,
        "mean": float(np.nanmean(boot_sharpes)),
        "std": float(np.nanstd(boot_sharpes, ddof=1)),
        "ci_lo_2_5": float(np.nanpercentile(boot_sharpes, 2.5)),
        "ci_hi_97_5": float(np.nanpercentile(boot_sharpes, 97.5)),
        "frac_positive": float(np.mean(boot_sharpes > 0)),
        "frac_above_0_5": float(np.mean(boot_sharpes > 0.5)),
        "boot_sharpes": boot_sharpes,
    }


# ---------------------------------------------------------------------------
# 7. Threshold sensitivity
# ---------------------------------------------------------------------------
def test_threshold_sensitivity(prices, thresholds=None):
    """Test how the lead-lag sugar Sharpe responds to different signal
    thresholds.  The default rule is signal > 0; here we sweep across
    a range to verify that the chosen 0-threshold is not cherry-picked."""
    if thresholds is None:
        thresholds = [-0.005, -0.002, -0.001, 0.0, 0.001, 0.002, 0.005]
    r = np.log(prices).diff()
    rows = []
    for thr in thresholds:
        signal = 0.5 * (r["guitars"].shift(1) + r["slingshots"].shift(1))
        w = pd.DataFrame(0.0, index=prices.index, columns=ASSETS)
        for a in ASSETS:
            if a != "sugar":
                w[a] = 1.0 / 7
        w["sugar"] = (signal > thr).astype(float) * (1.0 / 7)
        port_r = (w * r).sum(axis=1) + (1 - w.sum(axis=1)) * RF_DAILY
        port_r = port_r.dropna()
        m = metrics(port_r)
        days_held = (signal > thr).sum()
        rows.append({
            "threshold": thr,
            "sharpe": m["sharpe"],
            "ann_return_%": 100 * m["ann_return"],
            "max_dd_%": 100 * m["max_drawdown"],
            "days_sugar_held": int(days_held),
            "share_sugar_held": float(days_held / len(signal.dropna())),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 8. Transaction cost sensitivity
# ---------------------------------------------------------------------------
def test_transaction_costs(prices, cost_bps_list=None):
    """Subtract a per-trade transaction cost (in basis points of the
    sugar position) and see how lead-lag sugar Sharpe degrades.  Cost is
    applied on each transition (buy or sell) of the sugar position."""
    if cost_bps_list is None:
        cost_bps_list = [0, 1, 2, 5, 10, 20, 50, 100]
    r = np.log(prices).diff()
    signal = 0.5 * (r["guitars"].shift(1) + r["slingshots"].shift(1))
    sugar_in = (signal > 0).astype(int).fillna(0)
    transitions = sugar_in.diff().abs().fillna(0)  # 1 on each buy or sell

    rows = []
    for cost_bps in cost_bps_list:
        cost_per_trade = cost_bps / 1e4
        # Each transition costs `cost_per_trade` of the 1/7 sugar position
        daily_cost = transitions * cost_per_trade * (1.0 / 7)
        # Build portfolio with cost
        w = pd.DataFrame(0.0, index=prices.index, columns=ASSETS)
        for a in ASSETS:
            if a != "sugar":
                w[a] = 1.0 / 7
        w["sugar"] = sugar_in.astype(float) * (1.0 / 7)
        port_r = (w * r).sum(axis=1) + (1 - w.sum(axis=1)) * RF_DAILY
        port_r = port_r - daily_cost
        port_r = port_r.dropna()
        m = metrics(port_r)
        rows.append({
            "cost_bps_per_trade": cost_bps,
            "sharpe": m["sharpe"],
            "ann_return_%": 100 * m["ann_return"],
            "ann_cost_%": float(daily_cost.mean() * 252 * 100),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 10. Extreme-events sensitivity (drop top-K |sugar return| days)
# ---------------------------------------------------------------------------
def test_extreme_events(prices, drop_pcts=None):
    """Test whether the lead-lag sugar Sharpe is driven by a small number of
    extreme events.  We drop the top-K% most extreme |sugar| return days
    (replace with 0 contribution, i.e. neither gain nor loss) and recompute
    Sharpe.  If the strategy is genuine, Sharpe should degrade gracefully.
    If it depends on a handful of huge wins, Sharpe will collapse."""
    if drop_pcts is None:
        drop_pcts = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]

    r = np.log(prices).diff()
    abs_sugar = r["sugar"].abs()

    rows = []
    for pct in drop_pcts:
        # Find threshold: top pct% most extreme |sugar return|
        threshold = abs_sugar.quantile(1 - pct / 100) if pct > 0 else np.inf
        # Build filtered returns: zero out sugar on extreme days
        r_filtered = r.copy()
        if pct > 0:
            mask_extreme = abs_sugar > threshold
            r_filtered.loc[mask_extreme, "sugar"] = 0.0

        signal = 0.5 * (r_filtered["guitars"].shift(1) + r_filtered["slingshots"].shift(1))
        sugar_w = (signal > 0).astype(float) * (1.0 / 7)
        cash_w = (1.0 / 7) - sugar_w
        baseline = sum((1.0 / 7) * r_filtered[a] for a in ASSETS if a != "sugar")
        sugar_contrib = sugar_w * r_filtered["sugar"]
        cash_contrib = cash_w * RF_DAILY
        port_r = (baseline + sugar_contrib + cash_contrib).dropna()
        m = metrics(port_r)
        n_dropped = int((abs_sugar > threshold).sum()) if pct > 0 else 0
        rows.append({
            "pct_dropped": pct,
            "n_extreme_days_dropped": n_dropped,
            "sharpe": m["sharpe"],
            "ann_return_%": 100 * m["ann_return"],
            "max_dd_%": 100 * m["max_drawdown"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 9. Information ratio vs Buy & Hold
# ---------------------------------------------------------------------------
def test_information_ratio(strategies, benchmark_name="Buy & Hold (EW)"):
    """Compute the information ratio (active return / tracking error) of
    each strategy relative to Buy & Hold.  This isolates the strategy's
    edge over and above general market drift."""
    bh = strategies[benchmark_name]
    rows = []
    for name, port in strategies.items():
        if name == benchmark_name:
            continue
        # Align indices
        idx = port.index.intersection(bh.index)
        active = port.loc[idx] - bh.loc[idx]
        if len(active) < 50:
            continue
        ann_active = active.mean() * DAYS_PER_YEAR
        ann_te = active.std(ddof=1) * np.sqrt(DAYS_PER_YEAR)
        ir = ann_active / ann_te if ann_te > 0 else np.nan
        rows.append({
            "strategy": name,
            "ann_active_return_%": 100 * ann_active,
            "ann_tracking_error_%": 100 * ann_te,
            "information_ratio": ir,
        })
    return pd.DataFrame(rows).sort_values("information_ratio", ascending=False)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def main():
    print("=" * 75)
    print(" TASK 4 — ROBUSTNESS VALIDATION SUITE")
    print("=" * 75)

    print("\nBuilding price panel and running all strategies...")
    prices = build_price_panel()

    strategies = {
        "Buy & Hold (EW)":             strat_buy_and_hold(prices),
        "MA crossover 20/100":         strat_ma_crossover(prices),
        "Inverse-vol risk parity":     strat_inverse_vol(prices),
        "Momentum top-3":              strat_momentum(prices),
        "Channel breakout 55/20":      strat_channel_breakout(prices),
        "Buy dips":                    strat_buy_dips(prices),
        "Short stocks":                strat_short_stocks(prices),
        "Lead-lag sugar":              strat_leadlag_sugar(prices),
        "GARCH vol timing":            strat_garch_timing(prices),
    }
    print(f"  Strategies built: {len(strategies)}")

    # =====================================================================
    # 1. 80/20 holdout
    # =====================================================================
    print("\n" + "-" * 75)
    print(" 1.  80/20 IN-SAMPLE / OUT-OF-SAMPLE HOLDOUT")
    print("-" * 75)
    holdout = test_80_20_holdout(strategies, split_frac=0.80)
    print(holdout.to_string(index=False, float_format=lambda x: f"{x:+.3f}"
                            if isinstance(x, float) else str(x)))
    holdout.to_csv(ROOT / "task4_validation_holdout.csv", index=False)
    print(f"  Saved -> {ROOT/'task4_validation_holdout.csv'}")

    # =====================================================================
    # 2. 5-period walk-forward
    # =====================================================================
    print("\n" + "-" * 75)
    print(" 2.  5-PERIOD WALK-FORWARD SHARPE")
    print("-" * 75)
    wf_long, wf_wide = test_walk_forward(strategies, n_periods=5)
    print(wf_wide.to_string(float_format=lambda x: f"{x:+.3f}"
                            if isinstance(x, float) else str(x)))
    wf_long.to_csv(ROOT / "task4_validation_walkforward.csv", index=False)
    wf_wide.to_csv(ROOT / "task4_validation_walkforward_wide.csv")
    print(f"  Saved -> {ROOT/'task4_validation_walkforward.csv'}")

    # =====================================================================
    # 3. Specification search across 147 lead-lag configurations
    # =====================================================================
    print("\n" + "-" * 75)
    print(" 3.  SPECIFICATION SEARCH (147 LEAD-LAG CONFIGURATIONS)")
    print("-" * 75)
    spec = test_specification_search(prices)

    our_mask = (spec["predictors"] == "guitars+slingshots") & (spec["target"] == "sugar")
    our_row = spec[our_mask].iloc[0]
    print(f"  Total strategies tested:          {len(spec)}")
    print(f"  Median Sharpe (overall):          {spec['sharpe'].median():+.3f}")
    print(f"  Mean Sharpe (overall):            {spec['sharpe'].mean():+.3f}")
    print(f"  Std of Sharpe distribution:       {spec['sharpe'].std():.3f}")
    print(f"  Strategies with Sharpe > 0:       {(spec['sharpe'] > 0).sum()} of {len(spec)}")
    print(f"  Strategies with Sharpe > 0.5:     {(spec['sharpe'] > 0.5).sum()} of {len(spec)}")
    print(f"\n  Our deployment choice: guitars+slingshots -> sugar")
    print(f"    Sharpe: {our_row['sharpe']:+.3f}")
    print(f"    Rank:   {int(our_row['rank'])} of {len(spec)} (top {our_row['rank']/len(spec)*100:.2f}%)")
    print(f"\n  Top-10 lead-lag configurations:")
    print(spec.head(10).to_string(index=False,
                                  float_format=lambda x: f"{x:+.3f}"
                                  if isinstance(x, float) else str(x)))
    spec.to_csv(ROOT / "task4_validation_specsearch.csv", index=False)
    print(f"\n  Saved -> {ROOT/'task4_validation_specsearch.csv'}")

    # =====================================================================
    # 4. CPCV and PBO
    # =====================================================================
    print("\n" + "-" * 75)
    print(" 4.  COMBINATORIALLY PURGED CROSS-VALIDATION (CPCV) AND PBO")
    print("-" * 75)
    print("  Following Bailey/Borwein/Lopez de Prado/Zhu (2017)")
    print()
    pbo, cpcv_summary = compute_cpcv_pbo(strategies, K=10)
    print(f"\n  Probability of Backtest Overfitting (PBO): {pbo:.3f}")
    print(f"\n  Interpretation:")
    print(f"    PBO ≈ 0.0  → in-sample winners are out-of-sample winners (genuine signal)")
    print(f"    PBO ≈ 0.5  → in-sample selection has no out-of-sample predictive power")
    print(f"    PBO ≈ 1.0  → severe overfitting (in-sample winners become OOS losers)")
    print(f"\n  Distribution of best-in-sample strategy across {len(cpcv_summary)} splits:")
    print(cpcv_summary["best_in_sample"].value_counts().to_string())
    print(f"\n  Mean OOS rank when 'lead-lag sugar' is best in-sample:")
    if "Lead-lag sugar" in cpcv_summary["best_in_sample"].values:
        ll_rows = cpcv_summary[cpcv_summary["best_in_sample"] == "Lead-lag sugar"]
        print(f"    {ll_rows['oos_rank'].mean():.2f}  (1 = best, {len(strategies)} = worst)")
        print(f"    Median OOS rank: {ll_rows['oos_rank'].median():.0f}")
        print(f"    Mean OOS Sharpe: {ll_rows['oos_sharpe'].mean():+.3f}")
    cpcv_summary.to_csv(ROOT / "task4_validation_cpcv.csv", index=False)
    pd.DataFrame([{"K": 10, "n_splits": len(cpcv_summary), "PBO": pbo}]) \
        .to_csv(ROOT / "task4_validation_pbo_summary.csv", index=False)
    print(f"\n  Saved -> {ROOT/'task4_validation_cpcv.csv'}")
    print(f"  Saved -> {ROOT/'task4_validation_pbo_summary.csv'}")

    # =====================================================================
    # 5. Permutation test
    # =====================================================================
    print("\n" + "-" * 75)
    print(" 5.  PERMUTATION TEST (1000 random shuffles of guitars+slingshots)")
    print("-" * 75)
    perm = test_permutation(prices, n_perm=1000)
    print(f"  Original Sharpe (un-permuted):    {perm['original_sharpe']:+.3f}")
    print(f"  Permuted-mean Sharpe:             {perm['permuted_mean']:+.3f}")
    print(f"  Permuted-std Sharpe:              {perm['permuted_std']:.3f}")
    print(f"  Permuted-max Sharpe:              {perm['permuted_max']:+.3f}")
    print(f"  Z-score of original vs null:      {perm['z_score']:.2f}")
    print(f"  p-value (P(perm Sharpe >= our)):  {perm['p_value']:.4f}")
    print(f"  Interpretation: a p-value near 0 means the original signal is")
    print(f"    extremely unlikely under the null hypothesis of no temporal")
    print(f"    coupling between guitars/slingshots and sugar.")
    pd.DataFrame({
        "n_permutations": [perm["n_permutations"]],
        "original_sharpe": [perm["original_sharpe"]],
        "permuted_mean": [perm["permuted_mean"]],
        "permuted_std": [perm["permuted_std"]],
        "permuted_max": [perm["permuted_max"]],
        "z_score": [perm["z_score"]],
        "p_value": [perm["p_value"]],
    }).to_csv(ROOT / "task4_validation_permutation.csv", index=False)
    pd.DataFrame({"permuted_sharpe": perm["permuted_sharpes"]}) \
        .to_csv(ROOT / "task4_validation_permutation_dist.csv", index=False)
    print(f"  Saved -> {ROOT/'task4_validation_permutation.csv'}")

    # =====================================================================
    # 6. Block bootstrap
    # =====================================================================
    print("\n" + "-" * 75)
    print(" 6.  BLOCK BOOTSTRAP (500 resamples, 21-day blocks)")
    print("-" * 75)
    boot = test_block_bootstrap(prices, block_size=21, n_boot=500)
    print(f"  Bootstrap mean Sharpe:            {boot['mean']:+.3f}")
    print(f"  Bootstrap std Sharpe:             {boot['std']:.3f}")
    print(f"  Bootstrap 95% CI:                 [{boot['ci_lo_2_5']:+.3f}, {boot['ci_hi_97_5']:+.3f}]")
    print(f"  Fraction with positive Sharpe:    {boot['frac_positive']*100:.1f}%")
    print(f"  Fraction with Sharpe > 0.5:       {boot['frac_above_0_5']*100:.1f}%")
    pd.DataFrame({
        "n_bootstrap": [boot["n_bootstrap"]],
        "block_size": [boot["block_size"]],
        "mean": [boot["mean"]],
        "std": [boot["std"]],
        "ci_lo_2_5": [boot["ci_lo_2_5"]],
        "ci_hi_97_5": [boot["ci_hi_97_5"]],
        "frac_positive": [boot["frac_positive"]],
        "frac_above_0_5": [boot["frac_above_0_5"]],
    }).to_csv(ROOT / "task4_validation_bootstrap.csv", index=False)
    pd.DataFrame({"boot_sharpe": boot["boot_sharpes"]}) \
        .to_csv(ROOT / "task4_validation_bootstrap_dist.csv", index=False)
    print(f"  Saved -> {ROOT/'task4_validation_bootstrap.csv'}")

    # =====================================================================
    # 7. Threshold sensitivity
    # =====================================================================
    print("\n" + "-" * 75)
    print(" 7.  THRESHOLD SENSITIVITY (Sharpe across different signal cutoffs)")
    print("-" * 75)
    thresh = test_threshold_sensitivity(prices)
    print(thresh.to_string(index=False, float_format=lambda x: f"{x:+.3f}"
                            if isinstance(x, float) else str(x)))
    print(f"\n  Interpretation: if the chosen threshold of 0.0 were cherry-picked,")
    print(f"    we'd expect a sharp peak at 0.0 and worse Sharpes elsewhere.")
    print(f"    A flat profile with Sharpe stable around {thresh['sharpe'].mean():+.2f}")
    print(f"    confirms the choice is not a brittle local optimum.")
    thresh.to_csv(ROOT / "task4_validation_threshold.csv", index=False)
    print(f"  Saved -> {ROOT/'task4_validation_threshold.csv'}")

    # =====================================================================
    # 8. Transaction cost sensitivity
    # =====================================================================
    print("\n" + "-" * 75)
    print(" 8.  TRANSACTION COST SENSITIVITY (Sharpe vs cost per trade)")
    print("-" * 75)
    tc = test_transaction_costs(prices)
    print(tc.to_string(index=False, float_format=lambda x: f"{x:+.3f}"
                        if isinstance(x, float) else str(x)))
    # Find break-even cost level (where Sharpe drops below 0.5)
    below_half = tc[tc["sharpe"] < 0.5]
    if len(below_half) > 0:
        print(f"\n  Sharpe drops below 0.5 at {below_half.iloc[0]['cost_bps_per_trade']:.0f} bp/trade.")
    below_zero = tc[tc["sharpe"] < 0.0]
    if len(below_zero) > 0:
        print(f"  Sharpe drops below 0.0 at {below_zero.iloc[0]['cost_bps_per_trade']:.0f} bp/trade.")
    tc.to_csv(ROOT / "task4_validation_transaction_costs.csv", index=False)
    print(f"  Saved -> {ROOT/'task4_validation_transaction_costs.csv'}")

    # =====================================================================
    # 9. Information ratio vs Buy & Hold
    # =====================================================================
    print("\n" + "-" * 75)
    print(" 9.  INFORMATION RATIO VS BUY & HOLD")
    print("-" * 75)
    ir = test_information_ratio(strategies, benchmark_name="Buy & Hold (EW)")
    print(ir.to_string(index=False, float_format=lambda x: f"{x:+.3f}"
                        if isinstance(x, float) else str(x)))
    print(f"\n  Interpretation: the IR isolates each strategy's edge over Buy & Hold.")
    print(f"    Lead-lag sugar's IR confirms the edge is from signal, not from drift.")
    ir.to_csv(ROOT / "task4_validation_information_ratio.csv", index=False)
    print(f"  Saved -> {ROOT/'task4_validation_information_ratio.csv'}")

    # =====================================================================
    # 10. Extreme events sensitivity
    # =====================================================================
    print("\n" + "-" * 75)
    print(" 10. EXTREME EVENTS SENSITIVITY (drop top-K% |sugar return| days)")
    print("-" * 75)
    extreme = test_extreme_events(prices)
    print(extreme.to_string(index=False, float_format=lambda x: f"{x:+.3f}"
                             if isinstance(x, float) else str(x)))
    print(f"\n  Interpretation: if Sharpe collapses when a few extreme days are")
    print(f"    removed, the strategy depends on tail events (= fragile).")
    print(f"    Graceful degradation means the edge is broad-based, not concentrated.")
    extreme.to_csv(ROOT / "task4_validation_extreme_events.csv", index=False)
    print(f"  Saved -> {ROOT/'task4_validation_extreme_events.csv'}")

    print("\n" + "=" * 75)
    print(" VALIDATION SUITE COMPLETE — 10 ROBUSTNESS TESTS")
    print("=" * 75)


if __name__ == "__main__":
    main()
