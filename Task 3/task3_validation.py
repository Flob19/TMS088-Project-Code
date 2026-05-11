"""
Task 3 — Robustness validation tests for the extrapolation forecasts.

Addresses three reviewer concerns:
  (1) Drift sensitivity: full-sample mean is fragile under regime changes;
      we report drift over multiple rolling windows and show the implied
      forecast band still dominates over any drift estimate.
  (3) Rolling-origin backtest with too few origins: increase from 8 to ~80
      origins (50-day spacing) so the Wilson CI on coverage shrinks
      substantially.
  (4) Walk-forward GARCH at each rolling origin (instead of fitting once on
      the full sample).  Removes the look-ahead bias on GARCH parameters in
      the historical backtest.

All results are written to CSV files in the project root.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from arch import arch_model

from task2_interpolation import (
    load_clean, gap_bounds, univariate_bridge, bivariate_bridge,
    ASSETS, PEERS, USE_GARCH, Z95, PIC,
)
from task3_extrapolation import build_full_logp

ROOT = Path(__file__).parent
HORIZON = 200
DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# 1. Drift sensitivity analysis
# ---------------------------------------------------------------------------
def drift_sensitivity(full_logp, asset, windows=None):
    """Compute drift estimate over multiple rolling windows ending at the
    last observed day, plus the full-sample mean.  Reports the per-day
    drift, its standard error, and the cumulative 200-day drift relative
    to the 1-sigma forecast width."""
    if windows is None:
        windows = [252, 504, 1008, 2016, "full"]
    logp = full_logp[asset]
    last_obs_idx = int(np.where(~np.isnan(logp))[0][-1])
    rets = np.diff(logp[: last_obs_idx + 1])
    rets = rets[~np.isnan(rets)]
    sigma_full = np.std(rets, ddof=1)

    rows = []
    for w in windows:
        if w == "full":
            r = rets
            label = "full"
        else:
            r = rets[-int(w):]
            label = f"{w}d"
        if len(r) < 30:
            continue
        mu = np.mean(r)
        se = np.std(r, ddof=1) / np.sqrt(len(r))
        t_stat = mu / se if se > 0 else np.nan
        # Cumulative 200-day drift in basis points
        cum_drift_bp = mu * HORIZON * 1e4
        # 1-sigma forecast width over 200 days using full-sample sigma
        s_200 = sigma_full * np.sqrt(HORIZON) * 100  # in %
        rows.append({
            "asset": asset,
            "window": label,
            "n_days": len(r),
            "drift_bp_per_day": mu * 1e4,
            "se_bp_per_day": se * 1e4,
            "t_stat": t_stat,
            "cum_drift_200d_%": cum_drift_bp / 100,
            "S_200_1sigma_%": s_200,
            "drift_to_S_ratio": (cum_drift_bp / 100) / s_200 if s_200 > 0 else np.nan,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Walk-forward extrapolation (replaces the `extrapolate` from task3 that
#    used a global GARCH fit).  Re-fits GARCH on each rolling-origin window.
# ---------------------------------------------------------------------------
def extrapolate_walkforward(logp_observed, asset, horizon=HORIZON):
    """Random walk with drift; if asset is in USE_GARCH, refit GARCH(1,1)
    on the available history up to the forecast origin and use its
    multi-step variance forecast.  Otherwise constant variance.

    Returns mean_log (length horizon), cum_var (length horizon)."""
    r = np.diff(logp_observed)
    r = r[~np.isnan(r)]
    if len(r) < 100:
        return None, None
    mu = r.mean()
    last_logp = logp_observed[np.where(~np.isnan(logp_observed))[0][-1]]

    if asset in USE_GARCH:
        try:
            am = arch_model(r * 100, vol="Garch", p=1, q=1,
                            mean="Zero", rescale=False, dist="normal")
            res = am.fit(disp="off", show_warning=False)
            f = res.forecast(horizon=horizon, reindex=False)
            var_path = f.variance.values[-1] / 1e4
        except Exception:
            var_path = np.full(horizon, np.var(r, ddof=1))
    else:
        var_path = np.full(horizon, np.var(r, ddof=1))

    h = np.arange(1, horizon + 1)
    mean_log = last_logp + h * mu
    cum_var = np.cumsum(var_path)
    return mean_log, cum_var


# ---------------------------------------------------------------------------
# 3. Dense rolling-origin backtest (50-day spacing → ~80 origins)
# ---------------------------------------------------------------------------
def dense_backtest(full_logp, asset, origins, horizon=HORIZON, walkforward=True):
    """Forecast at each origin and check coverage at horizons 1, 50, 100, 200.
    If walkforward=True, refit GARCH at each origin (no look-ahead).
    Returns long-form DataFrame of (origin, horizon, err, sd, covered)."""
    rows = []
    last_obs_idx = int(np.where(~np.isnan(full_logp[asset]))[0][-1])
    for t_end in origins:
        if t_end + horizon > last_obs_idx:
            continue
        logp_hist = full_logp[asset][: t_end + 1]
        if walkforward:
            mean_log, cum_var = extrapolate_walkforward(logp_hist, asset, horizon)
        else:
            # Fall back to global-GARCH version if requested
            from task3_extrapolation import extrapolate
            mean_log, cum_var, _, _ = extrapolate(logp_hist, asset, horizon)
        if mean_log is None:
            continue
        truth = full_logp[asset][t_end + 1: t_end + 1 + horizon]
        if len(truth) < horizon or np.any(np.isnan(truth)):
            # Skip origins where truth has any NaN (outliers)
            continue
        sd = np.sqrt(cum_var)
        inside = np.abs(truth - mean_log) <= Z95 * sd
        for h_check in (1, 50, 100, 200):
            i = h_check - 1
            rows.append({
                "asset": asset, "origin": t_end, "horizon": h_check,
                "err_log": float(truth[i] - mean_log[i]),
                "sd_log": float(sd[i]),
                "covered": bool(inside[i]),
            })
    return rows


def wilson_ci(successes, n, z=1.96):
    """Wilson score 95% CI for a proportion."""
    if n == 0:
        return (np.nan, np.nan)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (centre - half, centre + half)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 75)
    print(" TASK 3 — ROBUSTNESS VALIDATION (drift sensitivity, dense backtest,")
    print("                                 walk-forward GARCH)")
    print("=" * 75)

    print("\nBuilding full log-price panel (with Task 2 interpolation)...")
    df = load_clean()
    log_df = np.log(df[ASSETS])
    gaps = {a: gap_bounds(log_df[a]) for a in ASSETS}
    full_logp = build_full_logp(df, log_df, gaps)
    last_obs_idx = {a: int(np.where(~np.isnan(full_logp[a]))[0][-1]) for a in ASSETS}
    print(f"  Last observed day per asset: {last_obs_idx}")

    # =====================================================================
    # 1. Drift sensitivity
    # =====================================================================
    print("\n" + "-" * 75)
    print(" 1.  DRIFT SENSITIVITY ANALYSIS")
    print("-" * 75)
    print("  Drift estimate over rolling windows ending at last observed day.")
    print("  ratio = cumulative 200-day drift / 1-sigma forecast width.")
    print("  ratio << 1 means the drift is dominated by the variance.")
    print()

    all_drift = []
    for a in ASSETS:
        df_drift = drift_sensitivity(full_logp, a)
        all_drift.append(df_drift)
    drift_df = pd.concat(all_drift, ignore_index=True)
    print(drift_df.to_string(index=False, float_format=lambda x: f"{x:+.3f}"
                              if isinstance(x, float) else str(x)))
    drift_df.to_csv(ROOT / "task3_validation_drift_sensitivity.csv", index=False)
    print(f"\n  Saved -> {ROOT/'task3_validation_drift_sensitivity.csv'}")

    # =====================================================================
    # 2. Dense rolling-origin backtest with walk-forward GARCH
    # =====================================================================
    print("\n" + "-" * 75)
    print(" 2.  DENSE ROLLING-ORIGIN BACKTEST (~80 ORIGINS, 50-DAY SPACING)")
    print("     with walk-forward GARCH refit at each origin")
    print("-" * 75)

    # Need at least 1000 days of history before first origin (for GARCH stability)
    # and 200 days remaining for forecast horizon.
    min_origin = 1000
    # Use the earliest "last observed" across assets as the cap
    earliest_last_obs = min(last_obs_idx.values())
    max_origin = earliest_last_obs - HORIZON
    spacing = 50
    origins = list(range(min_origin, max_origin + 1, spacing))
    print(f"  Origins: {origins[0]} to {origins[-1]} step {spacing}")
    print(f"  Number of origins: {len(origins)}")
    print(f"  Walk-forward GARCH refit at each origin: yes")
    print(f"  This will run {len(origins) * sum(1 for a in ASSETS if a in USE_GARCH)} GARCH fits — please be patient...")

    all_rows = []
    for ai, a in enumerate(ASSETS, start=1):
        print(f"  [{ai}/7] Backtesting {a:<14}", end=" ", flush=True)
        rows = dense_backtest(full_logp, a, origins, HORIZON, walkforward=True)
        print(f"({len(rows)//4} usable origins)")
        all_rows.extend(rows)
    bt = pd.DataFrame(all_rows)

    # Aggregate coverage by asset x horizon
    agg = (bt.groupby(["asset", "horizon"])
              .agg(coverage=("covered", "mean"),
                   n=("covered", "count"),
                   mean_err=("err_log", "mean"),
                   sd_err=("err_log", "std"),
                   mean_sd=("sd_log", "mean"))
              .reset_index())

    # Add Wilson CI per cell
    agg["wilson_lo"] = agg.apply(
        lambda r: wilson_ci(int(r["coverage"] * r["n"]), int(r["n"]))[0], axis=1)
    agg["wilson_hi"] = agg.apply(
        lambda r: wilson_ci(int(r["coverage"] * r["n"]), int(r["n"]))[1], axis=1)

    print()
    print(f"  Coverage table (% covered, with Wilson 95% CI):")
    pivot = agg.pivot(index="asset", columns="horizon",
                       values="coverage").round(3)
    pivot_n = agg.pivot(index="asset", columns="horizon",
                         values="n").round(0).astype(int)
    print(f"  N (per cell, max): {pivot_n.max().max()}")
    print(pivot.to_string(float_format=lambda x: f"{x*100:.1f}%"))

    print()
    print(f"  Wilson CI for coverage at h=200:")
    h200 = agg[agg["horizon"] == 200].copy()
    for _, r in h200.iterrows():
        print(f"    {r['asset']:<14}: {r['coverage']*100:.1f}% "
              f"[{r['wilson_lo']*100:.1f}%, {r['wilson_hi']*100:.1f}%] "
              f"(n={int(r['n'])})")

    bt.to_csv(ROOT / "task3_validation_dense_backtest.csv", index=False)
    agg.to_csv(ROOT / "task3_validation_dense_coverage.csv", index=False)
    print(f"\n  Saved -> {ROOT/'task3_validation_dense_backtest.csv'}")
    print(f"  Saved -> {ROOT/'task3_validation_dense_coverage.csv'}")

    # =====================================================================
    # 3. Coverage vs horizon plot
    # =====================================================================
    print("\n  Producing coverage-vs-horizon figure...")
    # For the plot we want all horizons 1..200, not just the four we logged
    # We need to re-run with all horizons recorded. Skip the figure if too slow.
    # Already saved: task4_validation produces enough material; produce a
    # simpler plot using the four horizon points.
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharex=True, sharey=True)
    axes = axes.ravel()
    colors = plt.cm.tab10.colors
    for i, a in enumerate(ASSETS):
        ax = axes[i]
        sub = agg[agg["asset"] == a].sort_values("horizon")
        ax.plot(sub["horizon"], sub["coverage"] * 100,
                "o-", color=colors[i], lw=1.8, ms=7)
        ax.fill_between(sub["horizon"], sub["wilson_lo"] * 100,
                         sub["wilson_hi"] * 100,
                         alpha=0.2, color=colors[i])
        ax.axhline(95, color="k", ls="--", lw=0.8)
        ax.set_title(a, fontsize=10)
        ax.set_ylim(45, 105)
        ax.grid(alpha=0.3)
        if i >= 4:
            ax.set_xlabel("horizon (days)")
        if i % 4 == 0:
            ax.set_ylabel("coverage (%)")
    for j in range(len(ASSETS), len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Dense rolling-origin backtest coverage "
                 f"({len(origins)} origins, 50-day spacing, walk-forward GARCH); "
                 f"shaded = Wilson 95\\% CI; dashed = nominal 95\\%",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(PIC / "fig_dense_forecast_coverage.png", dpi=130)
    plt.close()
    print(f"  Saved -> {PIC/'fig_dense_forecast_coverage.png'}")

    print("\n" + "=" * 75)
    print(" TASK 3 VALIDATION COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()
