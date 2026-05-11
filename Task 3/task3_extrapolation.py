"""
Task 3 — Extrapolation of the trailing 200 missing days for each Spiff series.

Method selection follows Task 1:
  * Random walk with drift on log-prices is the honest baseline for all seven
    series, since they are I(1) with stationary returns (ADF rejects a unit
    root on returns but not on prices) and the conditional mean is weakly
    predictable at best (Ljung-Box on returns).
  * For the four series with strong ARCH effects (slingshots, guitars, sugar,
    tranquillity) we augment the random walk with a GARCH(1,1) variance
    projection so the forecast band adapts to the volatility regime at the
    forecast origin.
  * Stocks / gurkor / water use a constant per-step variance estimated from
    the full cleaned-and-interpolated history.

For each asset we produce:
  - a point forecast on the price scale (median of log-normal)
  - a pointwise 95% band on the price scale
  - a small-Monte-Carlo set of paths for illustration
and validate by rolling-origin backtests at four past end-points.
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
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR

ROOT = Path(__file__).parent
RNG = np.random.default_rng(20260415)
HORIZON = 200


# ---------------------------------------------------------------------------
# 1. Build a fully observed log-price history (real data + interpolated gap)
# ---------------------------------------------------------------------------
def build_full_logp(df, log_df, gaps):
    """Return a dict asset -> numpy log-price array with the embedded 50-day
    gap filled in by the chosen Task 2 method.  The trailing 200 NaN remain
    NaN (that's what we forecast)."""
    out = {}
    for a in ASSETS:
        logp = log_df[a].to_numpy().copy()
        t_L, t_R = gaps[a]
        peer = PEERS.get(a)
        use_garch = a in USE_GARCH
        if peer is not None:
            peer_logp = log_df[peer].to_numpy()
            mu, _ = bivariate_bridge(logp, peer_logp, t_L, t_R, use_garch=use_garch)
        else:
            mu, _ = univariate_bridge(logp, t_L, t_R, use_garch=use_garch)
        logp[t_L + 1:t_R] = mu
        out[a] = logp
    return out


# ---------------------------------------------------------------------------
# 2. Random-walk-with-drift forecast, GARCH-augmented for USE_GARCH assets
# ---------------------------------------------------------------------------
def fit_garch_and_forecast_variance(returns, horizon):
    r = returns[~np.isnan(returns)]
    try:
        am = arch_model(r * 100, vol="Garch", p=1, q=1,
                        mean="Zero", rescale=False, dist="normal")
        res = am.fit(disp="off", show_warning=False)
        f = res.forecast(horizon=horizon, reindex=False)
        var_path = f.variance.values[-1] / 1e4  # undo *100 scale
        return np.asarray(var_path), res
    except Exception:
        sig2 = np.var(r, ddof=1)
        return np.full(horizon, sig2), None


def arima_forecast(logp_observed, horizon=HORIZON, order=(1, 1, 1)):
    """ARIMA(p,1,q) on log-prices; returns mean path + cumulative sd."""
    y = logp_observed[~np.isnan(logp_observed)]
    try:
        res = ARIMA(y, order=order, enforce_stationarity=False,
                    enforce_invertibility=False).fit(method_kwargs={"warn_convergence": False})
        fc = res.get_forecast(steps=horizon)
        mean = fc.predicted_mean
        sd = np.sqrt(fc.var_pred_mean)
        return np.asarray(mean), np.asarray(sd) ** 2  # return cumulative variance
    except Exception as exc:
        print(f"  ARIMA failed: {exc}")
        # fallback: RW with drift
        r = np.diff(y)
        mu = r.mean(); sig2 = np.var(r, ddof=1)
        h = np.arange(1, horizon + 1)
        return y[-1] + h * mu, h * sig2


def var_forecast_pair(logp_target, logp_peer, horizon=HORIZON):
    """VAR(1) on the two-asset log-price differences, returns forecast for
    the TARGET series only (mean path + cumulative variance)."""
    y1 = logp_target[~np.isnan(logp_target)]
    y2 = logp_peer[~np.isnan(logp_peer)]
    n = min(len(y1), len(y2))
    dY = np.column_stack([np.diff(y1[-n:]), np.diff(y2[-n:])])
    try:
        model = VAR(dY).fit(1)
        fc = model.forecast(dY[-model.k_ar:], steps=horizon)
        # cumulative mean forecast of target's differences = mean log-price path
        cum_mean = np.cumsum(fc[:, 0]) + y1[-1]
        # forecast variance from the VAR MSE
        mse = model.forecast_cov(steps=horizon)  # (horizon, 2, 2)
        cum_var = np.zeros(horizon)
        acc = 0.0
        for h in range(horizon):
            acc += mse[h][0, 0]
            cum_var[h] = acc
        return cum_mean, cum_var
    except Exception as exc:
        print(f"  VAR failed: {exc}")
        return None, None


def extrapolate(logp_observed, asset, horizon=HORIZON):
    """
    Return (mean_log, var_log) of length `horizon` for the trailing forecast.
    `logp_observed` is the log-price array with the embedded gap interpolated
    and the trailing NaNs trimmed off.
    """
    r = np.diff(logp_observed)
    r = r[~np.isnan(r)]
    mu = r.mean()
    last_logp = logp_observed[np.where(~np.isnan(logp_observed))[0][-1]]

    if asset in USE_GARCH:
        var_path, _ = fit_garch_and_forecast_variance(r, horizon)
    else:
        var_path = np.full(horizon, np.var(r, ddof=1))

    h = np.arange(1, horizon + 1)
    mean_log = last_logp + h * mu
    cum_var = np.cumsum(var_path)
    return mean_log, cum_var, mu, var_path


# ---------------------------------------------------------------------------
# 3. Rolling-origin backtest
# ---------------------------------------------------------------------------
def backtest(logp_full, asset, origins, horizon=HORIZON):
    """For each origin t_end, fit on logp_full[:t_end+1] (trailing inclusive)
    and compare the forecast against logp_full[t_end+1:t_end+1+horizon]."""
    rows = []
    for t_end in origins:
        logp_hist = logp_full[:t_end + 1]
        mean_log, cum_var, _, _ = extrapolate(logp_hist, asset, horizon)
        truth = logp_full[t_end + 1:t_end + 1 + horizon]
        if len(truth) < horizon:
            continue
        sd = np.sqrt(cum_var)
        inside = np.abs(truth - mean_log) <= Z95 * sd
        for h_check in (1, 50, 100, 200):
            i = h_check - 1
            rows.append({
                "asset": asset, "origin": t_end, "horizon": h_check,
                "err_log": truth[i] - mean_log[i],
                "sd_log": sd[i],
                "covered": bool(inside[i]),
            })
        # aggregate over horizon
        rmse = np.sqrt(np.mean((truth - mean_log) ** 2))
        cov = inside.mean()
        rows.append({
            "asset": asset, "origin": t_end, "horizon": "1-200 mean",
            "err_log": np.nan, "sd_log": np.nan,
            "covered": None,
            "rmse_log_avg": rmse, "coverage_avg": cov,
        })
    return rows


def pit_and_log_score(logp_full, asset, origins, horizon=HORIZON):
    """Probability Integral Transform values and Gaussian log-score for each
    (origin, horizon).  PIT_t,h = Phi((y_t+h - mu_t+h) / sigma_t+h).  If the
    predictive distribution is correctly calibrated, PIT values should be
    i.i.d. Uniform(0,1).  Log-score is the log predictive density at the
    realised value, averaged over folds; higher is better (sharper+calibrated)."""
    pits, logscores = [], []
    for t_end in origins:
        logp_hist = logp_full[:t_end + 1]
        mean_log, cum_var, _, _ = extrapolate(logp_hist, asset, horizon)
        truth = logp_full[t_end + 1:t_end + 1 + horizon]
        if len(truth) < horizon:
            continue
        sd = np.sqrt(cum_var)
        # skip the 5 isolated outlier-removal NaNs inside the truth window
        ok = ~np.isnan(truth)
        z = (truth[ok] - mean_log[ok]) / sd[ok]
        pits.extend(stats.norm.cdf(z).tolist())
        logscores.extend(stats.norm.logpdf(truth[ok], loc=mean_log[ok],
                                            scale=sd[ok]).tolist())
    return np.asarray(pits), np.asarray(logscores)


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------
def main():
    df = load_clean()
    log_df = np.log(df[ASSETS])
    gaps = {a: gap_bounds(log_df[a]) for a in ASSETS}
    full_logp = build_full_logp(df, log_df, gaps)

    # last observed index per asset (same for all: N - 201)
    N = len(df)
    last_obs_idx = {a: int(np.where(~np.isnan(full_logp[a]))[0][-1]) for a in ASSETS}
    print("Last observed day per asset:", last_obs_idx)

    # ----- produce trailing forecasts -----
    forecasts = {}
    for a in ASSETS:
        logp_hist = full_logp[a][:last_obs_idx[a] + 1]
        mean_log, cum_var, drift, var_path = extrapolate(logp_hist, a, HORIZON)
        sd_log = np.sqrt(cum_var)
        forecasts[a] = {
            "start_day": last_obs_idx[a] + 1,
            "mean_log": mean_log,
            "sd_log": sd_log,
            "drift": drift,
        }
        print(f"{a:14s}: drift = {drift*1e4:+.1f} bp/day,  "
              f"sd@h=1 = {sd_log[0]*100:.2f}%,  sd@h=200 = {sd_log[-1]*100:.2f}%")

    # ----- save per-day trailing forecast -----
    fc_rows = []
    for a in ASSETS:
        f = forecasts[a]
        for h, (m_, s_) in enumerate(zip(f["mean_log"], f["sd_log"]), start=1):
            fc_rows.append({
                "asset": a, "day": f["start_day"] + h - 1, "horizon": h,
                "median": float(np.exp(m_)),
                "lo95": float(np.exp(m_ - Z95 * s_)),
                "hi95": float(np.exp(m_ + Z95 * s_)),
                "sd_log": float(s_),
            })
    pd.DataFrame(fc_rows).to_csv(ROOT / "task3_forecasts.csv", index=False)
    print(f"Saved -> {ROOT/'task3_forecasts.csv'}")

    # ----- benchmark vs ARIMA / VAR at a few origins -----
    print("\nBenchmark: RW+drift vs ARIMA(1,1,1) vs VAR(1) on peer pair "
          "(RMSE on log-price, horizons 50 / 100 / 200)")
    bench_origins = [2500, 3500, 4500]
    bench_rows = []
    peer_map = {"gurkor": "water", "water": "gurkor",
                "slingshots": "guitars", "guitars": "slingshots"}
    for a in ASSETS:
        for t_end in bench_origins:
            logp_hist = full_logp[a][:t_end + 1]
            if len(logp_hist) < 300:
                continue
            truth = full_logp[a][t_end + 1:t_end + 1 + HORIZON]
            if len(truth) < HORIZON:
                continue
            # RW
            mean_rw, cum_var_rw, _, _ = extrapolate(logp_hist, a, HORIZON)
            # ARIMA(1,1,1)
            mean_arima, cum_var_arima = arima_forecast(logp_hist, HORIZON, order=(1, 1, 1))
            # VAR(1) if peer
            mean_var, cum_var_var = None, None
            if a in peer_map:
                peer_hist = full_logp[peer_map[a]][:t_end + 1]
                mean_var, cum_var_var = var_forecast_pair(logp_hist, peer_hist, HORIZON)
            for h in (50, 100, 200):
                i = h - 1
                row = {"asset": a, "origin": t_end, "horizon": h,
                       "rw_err": truth[i] - mean_rw[i],
                       "arima_err": truth[i] - mean_arima[i]}
                if mean_var is not None:
                    row["var_err"] = truth[i] - mean_var[i]
                bench_rows.append(row)
    bench = pd.DataFrame(bench_rows)
    bench_agg = (bench.groupby(["asset", "horizon"]).agg(
        rw_rmse=("rw_err", lambda s: float(np.sqrt(np.mean(s**2)))),
        arima_rmse=("arima_err", lambda s: float(np.sqrt(np.mean(s**2)))),
        var_rmse=("var_err", lambda s: float(np.sqrt(np.nanmean(s**2))) if s.notna().any() else np.nan),
    ).reset_index())
    print(bench_agg.to_string(index=False))
    bench_agg.to_csv(ROOT / "task3_benchmark.csv", index=False)
    print(f"Saved -> {ROOT/'task3_benchmark.csv'}")

    # ----- rolling-origin backtest -----
    print("\nRolling-origin backtest (coverage @ nominal 95%):")
    origins_base = [1800, 2200, 2600, 3000, 3400, 3800, 4200, 4600]
    all_rows = []
    for a in ASSETS:
        rows = backtest(full_logp[a], a, origins_base, HORIZON)
        all_rows.extend(rows)
    bt = pd.DataFrame(all_rows)
    # aggregate by asset x horizon
    agg = (bt[bt["horizon"].apply(lambda x: isinstance(x, int))]
           .groupby(["asset", "horizon"])
           .agg(mean_err=("err_log", "mean"),
                sd_err=("err_log", "std"),
                mean_sd=("sd_log", "mean"),
                coverage=("covered", "mean"),
                n=("covered", "count"))
           .reset_index())
    print(agg.to_string(index=False))
    agg.to_csv(ROOT / "task3_backtest.csv", index=False)
    print(f"Saved -> {ROOT/'task3_backtest.csv'}")

    # ----- figure: trailing forecast per asset -----
    fig, axes = plt.subplots(4, 2, figsize=(13, 14))
    axes = axes.ravel()
    for ax, a in zip(axes, ASSETS):
        f = forecasts[a]
        t_end = f["start_day"] - 1
        lo_w = max(0, t_end - 300)
        prices = df[a].to_numpy()
        # fill in interpolated gap on the plot too, to show continuity
        logp_full = full_logp[a]
        prices_filled = np.exp(logp_full)
        ax.plot(np.arange(lo_w, t_end + 1), prices_filled[lo_w:t_end + 1],
                "k-", lw=0.7, label="history (inc. interpolated gap)")
        h_days = np.arange(f["start_day"], f["start_day"] + HORIZON)
        median = np.exp(f["mean_log"])
        lo95 = np.exp(f["mean_log"] - Z95 * f["sd_log"])
        hi95 = np.exp(f["mean_log"] + Z95 * f["sd_log"])
        ax.fill_between(h_days, lo95, hi95, color="C1", alpha=0.3, label="95% band")
        ax.plot(h_days, median, "C1-", lw=1.4, label="median forecast")
        ax.axvline(t_end, color="C3", ls="--", lw=0.8)
        ax.set_title(f"{a}  —  drift {f['drift']*1e4:+.1f} bp/day,  band @ h=200: "
                     f"$\\pm${Z95*f['sd_log'][-1]*100:.1f}%")
        ax.set_xlabel("day"); ax.set_ylabel("price")
        ax.legend(fontsize=7, loc="upper left")
    axes[-1].axis("off")
    plt.tight_layout()
    plt.savefig(PIC / "fig_forecast_paths.png", dpi=130)
    plt.close()
    print(f"Saved -> {PIC/'fig_forecast_paths.png'}")

    # ----- PIT and log-score diagnostics -----
    print("\nPIT and log-score diagnostics:")
    pit_rows = []
    all_pits = {}
    for a in ASSETS:
        pits, ls = pit_and_log_score(full_logp[a], a, origins_base, HORIZON)
        all_pits[a] = pits
        # KS test vs Uniform
        ks_stat, ks_p = stats.kstest(pits, "uniform")
        pit_rows.append({
            "asset": a, "n": len(pits),
            "ks_stat": float(ks_stat), "ks_pvalue": float(ks_p),
            "mean_log_score": float(ls.mean()),
        })
        print(f"  {a:14s}  KS vs U(0,1): stat={ks_stat:.3f} p={ks_p:.3g}  "
              f"mean log-score = {ls.mean():+.3f}")
    pd.DataFrame(pit_rows).to_csv(ROOT / "task3_pit.csv", index=False)

    # ----- figure: PIT histograms (small multiples) -----
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharey=True)
    axes = axes.ravel()
    for i, a in enumerate(ASSETS):
        ax = axes[i]
        ax.hist(all_pits[a], bins=10, range=(0, 1),
                color=plt.cm.tab10.colors[i], alpha=0.7, edgecolor="k")
        ax.axhline(len(all_pits[a]) / 10, color="k", ls="--", lw=0.8,
                   label="uniform")
        ax.set_title(a, fontsize=10)
        ax.set_xlim(0, 1)
        if i % 4 == 0:
            ax.set_ylabel("count")
        if i >= 4:
            ax.set_xlabel("PIT value")
    for j in range(len(ASSETS), len(axes)):
        axes[j].axis("off")
    fig.suptitle("PIT histograms across 8 origins $\\times$ 200 horizons "
                 "(uniform = well-calibrated predictive distribution)",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(PIC / "fig_pit_histograms.png", dpi=130)
    plt.close()
    print(f"Saved -> {PIC/'fig_pit_histograms.png'}")

    # ----- figure: GARCH variance path vs constant variance (cumulative) -----
    fig, ax = plt.subplots(figsize=(9, 5))
    h_grid = np.arange(1, HORIZON + 1)
    garch_assets_local = ["slingshots", "guitars", "sugar", "tranquillity"]
    for a in garch_assets_local:
        logp_hist = full_logp[a][:last_obs_idx[a] + 1]
        r = np.diff(logp_hist)
        r = r[~np.isnan(r)]
        # constant variance
        sig2_const = np.var(r, ddof=1)
        cum_const = h_grid * sig2_const
        # GARCH variance path
        var_path, _ = fit_garch_and_forecast_variance(r, HORIZON)
        cum_garch = np.cumsum(var_path)
        # plot ratio GARCH/const
        ax.plot(h_grid, cum_garch / cum_const, lw=1.5, label=a)
    ax.axhline(1, color="k", lw=0.7, ls="--",
               label="constant-$\\sigma$ baseline")
    ax.set_xlabel("forecast horizon $h$ (days)")
    ax.set_ylabel(r"$\sum_{j=1}^{h}\hat\sigma_{T+j}^2\,/\,h\hat\sigma^2$")
    ax.set_title("GARCH cumulative forecast variance relative to constant $\\sigma$")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PIC / "fig_garch_vs_const.png", dpi=130)
    plt.close()
    print(f"Saved -> {PIC/'fig_garch_vs_const.png'}")

    # ----- figure: backtest coverage as small multiples -----
    ncols = 4; nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 6),
                             sharex=True, sharey=True)
    axes = axes.ravel()
    colors = plt.cm.tab10.colors
    for i, a in enumerate(ASSETS):
        ax = axes[i]
        sub = agg[agg["asset"] == a].sort_values("horizon")
        ax.plot(sub["horizon"], sub["coverage"] * 100,
                "o-", color=colors[i], lw=1.8, ms=7)
        ax.axhline(95, color="k", ls="--", lw=0.8)
        ax.set_title(a, fontsize=10)
        ax.set_ylim(45, 105)
        ax.grid(alpha=0.3)
        if i >= ncols * (nrows - 1):
            ax.set_xlabel("horizon (days)")
        if i % ncols == 0:
            ax.set_ylabel("coverage (%)")
    for j in range(len(ASSETS), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Rolling-origin backtest coverage (8 origins per asset); "
                 "dashed line = nominal 95\\%",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(PIC / "fig_forecast_coverage.png", dpi=130)
    plt.close()
    print(f"Saved -> {PIC/'fig_forecast_coverage.png'}")


if __name__ == "__main__":
    main()
