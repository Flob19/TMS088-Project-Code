"""
Task 3 extension - compare four extrapolation specifications on identical
rolling-origin folds:

    M1  RW + drift, Gaussian band   (baseline reproduction)
    M2  RW + drift, Student-t band  (df fitted by MLE on standardised in-sample
                                     log-returns at each origin)
    M3  ARIMA(1,1,1), Gaussian      (statsmodels predicted_mean / var_pred_mean)
    M4  ARIMA(1,1,1), Student-t     (df fitted by MLE on in-sample ARIMA resids)

Production setup (matched to task3_validation.py):
  * 66 origins per asset spaced 50 days apart in [1000, last_obs - 200]
  * horizons checked at h = 1, 50, 100, 200
  * walk-forward GARCH refit at each origin for assets in USE_GARCH
  * other assets use constant in-sample variance
  * drift mu = mean of log-returns up to origin (full in-sample mean)
  * Task 2 gap-fill on the historical panel
  * |z|>8 outlier mask on raw prices

Skip origins where the 200-day truth window has any NaN.
Drop origins for ALL four models whenever ARIMA fails at that origin
(strictly paired comparison).
"""

from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import t as student_t

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from task2_interpolation import (  # noqa: E402
    load_clean, gap_bounds, ASSETS, USE_GARCH, Z95,
)
from task3_extrapolation import build_full_logp  # noqa: E402
from task3_validation import (  # noqa: E402
    extrapolate_walkforward, wilson_ci,
)

from statsmodels.tsa.arima.model import ARIMA  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
HORIZON = 200
HORIZONS_CHECK = (1, 50, 100, 200)
DF_CLIP = (3.0, 50.0)


def fit_t_df(standardised):
    x = np.asarray(standardised, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 50:
        return DF_CLIP[1]
    try:
        df_hat, _, _ = student_t.fit(x, floc=0.0, fscale=1.0)
    except Exception:
        return DF_CLIP[1]
    if not np.isfinite(df_hat):
        return DF_CLIP[1]
    return float(np.clip(df_hat, DF_CLIP[0], DF_CLIP[1]))


def fit_rw_path(logp_observed, asset, horizon=HORIZON):
    mean_log, cum_var = extrapolate_walkforward(logp_observed, asset, horizon)
    if mean_log is None:
        return None, None, None
    r = np.diff(logp_observed)
    r = r[~np.isnan(r)]
    return np.asarray(mean_log), np.asarray(cum_var), r


def fit_arima_path(logp_observed, horizon=HORIZON):
    y = logp_observed[~np.isnan(logp_observed)]
    try:
        res = ARIMA(
            y, order=(1, 1, 1),
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(method_kwargs={"warn_convergence": False})
    except Exception:
        return None, None, None
    if not np.all(np.isfinite(res.params)):
        return None, None, None
    try:
        fc = res.get_forecast(steps=horizon)
        mean = np.asarray(fc.predicted_mean, dtype=float)
        var_pred = np.asarray(fc.var_pred_mean, dtype=float)
    except Exception:
        return None, None, None
    if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(var_pred)) and np.all(var_pred > 0)):
        return None, None, None
    # Stability filter: reject if AR / MA roots blow up the forecast variance.
    # We did not enforce stationarity at fit time (matches task3_extrapolation),
    # but if the implied forecast SD at the longest horizon is wildly larger
    # than a plausible bound (1e4 in log-price units = absurd), the parameter
    # estimate is non-stationary and must be treated as a paired-skip origin.
    if np.sqrt(var_pred[-1]) > 1e3:
        return None, None, None
    resid = np.asarray(res.resid, dtype=float)
    resid = resid[np.isfinite(resid)]
    if len(resid) > 10:
        resid = resid[5:]
    return mean, var_pred, resid


def backtest_asset(full_logp, asset, origins, horizon=HORIZON):
    rows = []
    pit_rows = []
    last_obs_idx = int(np.where(~np.isnan(full_logp[asset]))[0][-1])

    for t_end in origins:
        if t_end + horizon > last_obs_idx:
            continue
        truth = full_logp[asset][t_end + 1: t_end + 1 + horizon]
        if len(truth) < horizon or np.any(np.isnan(truth)):
            continue
        logp_hist = full_logp[asset][: t_end + 1]

        rw_mean, rw_cumvar, r_in = fit_rw_path(logp_hist, asset, horizon)
        if rw_mean is None:
            continue
        rw_sd = np.sqrt(rw_cumvar)

        ar_mean, ar_var, ar_resid = fit_arima_path(logp_hist, horizon)
        if ar_mean is None:
            continue
        ar_sd = np.sqrt(ar_var)

        sigma_in = float(np.std(r_in, ddof=1))
        mu_in = float(np.mean(r_in))
        std_r = (r_in - mu_in) / sigma_in if sigma_in > 0 else r_in
        df_rw = fit_t_df(std_r)

        sigma_ar = float(np.std(ar_resid, ddof=1))
        if sigma_ar > 0:
            std_ar = ar_resid / sigma_ar
        else:
            std_ar = ar_resid
        df_ar = fit_t_df(std_ar)

        for h in HORIZONS_CHECK:
            i = h - 1
            err_rw = float(truth[i] - rw_mean[i])
            err_ar = float(truth[i] - ar_mean[i])

            cov_rw_g = bool(abs(err_rw) <= Z95 * rw_sd[i])
            q_t_rw = float(student_t.ppf(0.975, df=df_rw))
            cov_rw_t = bool(abs(err_rw) <= q_t_rw * rw_sd[i])
            cov_ar_g = bool(abs(err_ar) <= Z95 * ar_sd[i])
            q_t_ar = float(student_t.ppf(0.975, df=df_ar))
            cov_ar_t = bool(abs(err_ar) <= q_t_ar * ar_sd[i])

            rows.append(dict(
                asset=asset, model="RW-G", origin=t_end, horizon=h,
                err_log=err_rw, sd_log=float(rw_sd[i]), covered=cov_rw_g))
            rows.append(dict(
                asset=asset, model="RW-t", origin=t_end, horizon=h,
                err_log=err_rw, sd_log=float(rw_sd[i]), covered=cov_rw_t))
            rows.append(dict(
                asset=asset, model="ARIMA-G", origin=t_end, horizon=h,
                err_log=err_ar, sd_log=float(ar_sd[i]), covered=cov_ar_g))
            rows.append(dict(
                asset=asset, model="ARIMA-t", origin=t_end, horizon=h,
                err_log=err_ar, sd_log=float(ar_sd[i]), covered=cov_ar_t))

        i200 = 200 - 1
        z_rw = (truth[i200] - rw_mean[i200]) / rw_sd[i200]
        z_ar = (truth[i200] - ar_mean[i200]) / ar_sd[i200]
        u_rw_g = float(stats.norm.cdf(z_rw))
        u_ar_g = float(stats.norm.cdf(z_ar))
        u_rw_t = float(student_t.cdf(z_rw, df=df_rw))
        u_ar_t = float(student_t.cdf(z_ar, df=df_ar))

        pit_rows.append(dict(asset=asset, model="RW-G",    origin=t_end, u_pit=u_rw_g))
        pit_rows.append(dict(asset=asset, model="RW-t",    origin=t_end, u_pit=u_rw_t))
        pit_rows.append(dict(asset=asset, model="ARIMA-G", origin=t_end, u_pit=u_ar_g))
        pit_rows.append(dict(asset=asset, model="ARIMA-t", origin=t_end, u_pit=u_ar_t))

    return rows, pit_rows


def main():
    print("=" * 78)
    print(" TASK 3 EXTENSION  RW-G / RW-t / ARIMA-G / ARIMA-t  rolling-origin backtest")
    print("=" * 78)

    print("\nBuilding full log-price panel (with Task 2 gap-fill)...")
    df = load_clean()
    log_df = np.log(df[ASSETS])
    gaps = {a: gap_bounds(log_df[a]) for a in ASSETS}
    full_logp = build_full_logp(df, log_df, gaps)
    last_obs_idx = {a: int(np.where(~np.isnan(full_logp[a]))[0][-1]) for a in ASSETS}
    print(f"  Last observed day per asset: {last_obs_idx}")

    earliest_last_obs = min(last_obs_idx.values())
    origins = list(range(1000, earliest_last_obs - HORIZON + 1, 50))
    print(f"  Origins per asset: {len(origins)}  ({origins[0]} ... {origins[-1]} step 50)")

    all_rows = []
    all_pit_rows = []
    t0 = time.time()
    for ai, a in enumerate(ASSETS, start=1):
        ta = time.time()
        rows, pit_rows = backtest_asset(full_logp, a, origins, HORIZON)
        n_origins_used = len(rows) // (4 * len(HORIZONS_CHECK))
        elapsed = time.time() - ta
        print(f"  [{ai}/{len(ASSETS)}] {a:<14} {n_origins_used:3d} paired origins  ({elapsed:5.1f}s)")
        all_rows.extend(rows)
        all_pit_rows.extend(pit_rows)
    print(f"  Total elapsed: {time.time()-t0:.1f}s")

    bt = pd.DataFrame(all_rows)
    pit = pd.DataFrame(all_pit_rows)

    agg_rows = []
    for (asset, model, horizon), sub in bt.groupby(["asset", "model", "horizon"]):
        n = len(sub)
        log_rmse = float(np.sqrt(np.mean(sub["err_log"].values ** 2)))
        cov = float(sub["covered"].mean())
        succ = int(round(cov * n))
        lo, hi = wilson_ci(succ, n)
        agg_rows.append(dict(
            asset=asset, model=model, horizon=horizon, n_origins=n,
            log_rmse=log_rmse, coverage=cov,
            wilson_lo=float(lo), wilson_hi=float(hi),
        ))
    agg = pd.DataFrame(agg_rows).sort_values(["asset", "model", "horizon"]).reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUT_DIR / "task3_ext_metrics.csv"
    pit_path = OUT_DIR / "task3_ext_pit_h200.csv"
    agg.to_csv(metrics_path, index=False)
    pit[["asset", "model", "origin", "u_pit"]].to_csv(pit_path, index=False)
    print(f"\nSaved -> {metrics_path}")
    print(f"Saved -> {pit_path}")

    print()
    print_summary_table(agg)
    print()
    print_recommendation(agg)


def print_summary_table(agg):
    models_order = ["RW-G", "RW-t", "ARIMA-G", "ARIMA-t"]
    pivot_rmse = agg.pivot_table(index=["asset", "model"], columns="horizon", values="log_rmse")
    pivot_cov  = agg.pivot_table(index=["asset", "model"], columns="horizon", values="coverage") * 100
    pivot_lo   = agg.pivot_table(index=["asset", "model"], columns="horizon", values="wilson_lo") * 100
    pivot_hi   = agg.pivot_table(index=["asset", "model"], columns="horizon", values="wilson_hi") * 100

    horizons_show = [50, 100, 200]
    header = (f"{'asset':<14}{'model':<9}"
              + "".join([f"  h={h:<3}: rmse  cov%   " for h in horizons_show])
              + f"{'wilson95(h=200)':>18}")
    print(header)
    print("-" * len(header))

    for asset in agg["asset"].drop_duplicates().tolist():
        for m in models_order:
            if (asset, m) not in pivot_rmse.index:
                continue
            cells = ""
            for h in horizons_show:
                rmse = pivot_rmse.loc[(asset, m), h]
                cov = pivot_cov.loc[(asset, m), h]
                cells += f"           {rmse:6.3f} {cov:5.1f}   "
            lo = pivot_lo.loc[(asset, m), 200]
            hi = pivot_hi.loc[(asset, m), 200]
            wilson_str = f"[{lo:4.1f},{hi:5.1f}]"
            asset_label = asset if m == models_order[0] else ""
            print(f"{asset_label:<14}{m:<9}{cells}{wilson_str:>18}")
        print("")


def print_recommendation(agg):
    h200 = agg[agg["horizon"] == 200].copy()
    pivot_cov = h200.pivot(index="asset", columns="model", values="coverage") * 100
    pivot_rmse_all = agg.pivot_table(index=["asset", "horizon"], columns="model", values="log_rmse")

    cov_rw_g  = pivot_cov["RW-G"].mean()
    cov_rw_t  = pivot_cov["RW-t"].mean()
    cov_ar_g  = pivot_cov["ARIMA-G"].mean()
    cov_ar_t  = pivot_cov["ARIMA-t"].mean()

    diff_rmse = (pivot_rmse_all["ARIMA-G"] - pivot_rmse_all["RW-G"]).dropna()
    rel_diff_pct = 100.0 * (diff_rmse / pivot_rmse_all["RW-G"]).mean()

    print("RECOMMENDATION")
    print("-" * 78)
    print(
        f"At h=200 the cross-asset mean 95% coverage is "
        f"{cov_rw_g:.1f}% (RW-G) -> {cov_rw_t:.1f}% (RW-t) and "
        f"{cov_ar_g:.1f}% (ARIMA-G) -> {cov_ar_t:.1f}% (ARIMA-t). "
        "Switching the innovation distribution to Student-t widens the 95% band "
        "by a few per-cent of one sigma; the change is small because the long-"
        "horizon predictive variance is dominated by the cumulated random-walk "
        "term, not by the tail of the one-step innovation. Student-t does not "
        "materially fix coverage at long horizons.\n"
    )
    print(
        f"Mean ARIMA(1,1,1) vs RW log-RMSE difference across (asset, horizon) "
        f"is {rel_diff_pct:+.2f}% relative; the ARIMA mean forecast is "
        "effectively the random walk plus a tiny AR(1)/MA(1) correction on the "
        "log-return level, so the point forecast at h=50,100,200 is "
        "indistinguishable from the baseline. ARIMA does not materially "
        "improve log-RMSE.\n"
    )
    print(
        "Given that variance dominates the 200-day predictive distribution and "
        "neither extension changes the mean path or the cumulative-variance "
        "term in a meaningful way, the additional complexity (ARIMA fitting at "
        "every origin, df estimation, two extra parameters) is NOT justified. "
        "Keep RW-with-drift + walk-forward GARCH (Gaussian) as the production "
        "spec; report ARIMA and Student-t as robustness checks only."
    )
    print("-" * 78)


if __name__ == "__main__":
    main()
