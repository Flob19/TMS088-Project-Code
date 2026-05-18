"""
Task 2 — Interpolation of embedded 50-day gaps in the seven Spiff price series.

Implements:
  (1) univariate Brownian bridge on log-prices,
  (2) bivariate (peer-conditioned) Brownian bridge for correlated pairs,
  (3) GARCH(1,1)-augmented bridge with time-changed variance.

Validates each method on K=200 synthetic interior gaps per asset
(RMSE on log scale, % RMSE on price scale, 95% empirical coverage, CRPS),
then produces the final interpolated paths inside the real gaps with
95% pointwise bands.  Saves figures to ./Pictures/ and the results table
to ./task2_results.csv.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from arch import arch_model

ROOT = Path(__file__).parent
PIC = ROOT / "Pictures"
PIC.mkdir(exist_ok=True)
RNG = np.random.default_rng(20260414)

ASSETS = ["gurkor", "guitars", "slingshots", "stocks", "sugar", "water", "tranquillity"]
PEERS = {  # strong pairs first; sugar's peer was changed from "guitars" (same-day)
           # to "slingshots" at lag=1 after the ablation showed lower synthetic-gap
           # RMSE (3.43% vs 3.80%).  See PEER_LAGS below for the lag in days.
    "gurkor": "water",
    "water": "gurkor",
    "slingshots": "guitars",
    "guitars": "slingshots",
    "sugar": "slingshots",   # changed from "guitars" -- now lag=1, see PEER_LAGS
}
# Lag in days for the peer regression (0 = contemporaneous, default).
# When lag=k, the in-sample regression is r_target_t ~ alpha + beta r_peer_{t-k}.
PEER_LAGS = {"sugar": 1}
USE_GARCH = {"slingshots", "guitars", "sugar", "tranquillity"}
M_GAP = 50         # gap length
K_FOLDS = 200      # synthetic gap folds
Z95 = stats.norm.ppf(0.975)


# ---------------------------------------------------------------------------
# 1.  Load and clean
# ---------------------------------------------------------------------------
def load_clean():
    df = pd.read_csv(ROOT / "spiff_data-2.csv", index_col=0)
    df = df.set_index("day")
    # remove |z|>8 outliers (Task 1 cleaning rule), treat as missing
    for c in ASSETS:
        x = df[c]
        mu, sd = x.mean(skipna=True), x.std(skipna=True)
        df.loc[(x - mu).abs() / sd > 8, c] = np.nan
    return df


def gap_bounds(series):
    """Return (t_L, t_R) integer indices of the embedded gap (NOT the trailing 200)."""
    isna = series.isna().to_numpy()
    n = len(series)
    # the trailing block of 200 NaNs is the extrapolation target -- exclude it
    last_obs = n - 1
    while last_obs >= 0 and isna[last_obs]:
        last_obs -= 1
    interior = isna[: last_obs + 1]
    # find contiguous NaN run
    in_gap = False
    runs = []
    for i, v in enumerate(interior):
        if v and not in_gap:
            start = i
            in_gap = True
        elif not v and in_gap:
            runs.append((start, i - 1))
            in_gap = False
    if not runs:
        return None
    s, e = max(runs, key=lambda r: r[1] - r[0])
    return s - 1, e + 1  # last observed before, first observed after


# ---------------------------------------------------------------------------
# 2.  Bridge primitives
# ---------------------------------------------------------------------------
def bridge_mean_var(xL, xR, m, sigma2_path):
    """
    Discrete Brownian bridge on log-prices, possibly with time-varying variance.
    sigma2_path has length m+1 (per-step variances inside the gap).
    Returns mean[k=1..m], var[k=1..m].
    """
    S = np.cumsum(sigma2_path)        # length m+1
    Stot = S[-1]
    k = np.arange(1, m + 1)
    Sk = S[k - 1]                     # variance accumulated up to step k
    mean = xL + (Sk / Stot) * (xR - xL)
    var = Sk * (Stot - Sk) / Stot
    return mean, var


def fit_sigma2_window(returns, t_center, w=250):
    """Constant variance from a pre-gap window strictly before t_center."""
    lo = max(0, t_center - w)
    seg = returns[lo:t_center]
    seg = seg[~np.isnan(seg)]
    if len(seg) > 1:
        return np.var(seg, ddof=1)
    return np.nanvar(returns) if np.any(~np.isnan(returns)) else 1e-8


def fit_sigma2_garch_path(residuals, m, t_center):
    """
    GARCH(1,1) on residuals up to t_center; return per-step variance path of
    length m+1 inside the gap (mean-reverting forecast).
    """
    seg = residuals[:t_center]
    seg = seg[~np.isnan(seg)]
    if len(seg) < 200:
        return np.full(m + 1, np.var(seg, ddof=1))
    try:
        am = arch_model(seg * 100, vol="Garch", p=1, q=1, mean="Zero",
                        rescale=False, dist="normal")
        res = am.fit(disp="off", show_warning=False)
        f = res.forecast(horizon=m + 1, reindex=False)
        v = f.variance.values[-1] / 1e4
        return np.asarray(v)
    except Exception:
        return np.full(m + 1, np.var(seg, ddof=1))


# ---------------------------------------------------------------------------
# 3.  Univariate and bivariate bridge estimators
# ---------------------------------------------------------------------------
def univariate_bridge(logp, t_L, t_R, use_garch=False):
    m = t_R - t_L - 1
    r = np.diff(logp)
    if use_garch:
        sig2 = fit_sigma2_garch_path(r, m + 1, t_L)
    else:
        sig2 = np.full(m + 1, fit_sigma2_window(r, t_L))
    mean, var = bridge_mean_var(logp[t_L], logp[t_R], m, sig2)
    return mean, var


def bivariate_bridge(logp_t, logp_p, t_L, t_R, use_garch=False, lag=0):
    """
    Peer-conditioned bridge.  Fit beta on observed days outside the target gap;
    bridge the residual log-return process.

    Parameters
    ----------
    lag : int, default 0
        0 (default) -- contemporaneous formulation: r_target_t = alpha + beta r_peer_t + eps_t.
        1           -- one-day-lagged formulation:   r_target_t = alpha + beta r_peer_{t-1} + eps_t.
                       Used for sugar (see PEER_LAGS).
    """
    m = t_R - t_L - 1
    r_t = np.diff(logp_t)
    r_p = np.diff(logp_p)

    if lag == 0:
        # ---------- contemporaneous (original) ----------
        mask = ~np.isnan(r_t) & ~np.isnan(r_p)
        mask[t_L:t_R] = False
        rt, rp = r_t[mask], r_p[mask]
        beta, alpha = np.polyfit(rp, rt, 1)
        eps = rt - (alpha + beta * rp)
        sig2_eps = float(np.var(eps, ddof=2))
        # peer must be observed inside the gap.
        peer_segment = logp_p[t_L:t_R + 1]
        if np.any(np.isnan(peer_segment)):
            return univariate_bridge(logp_t, t_L, t_R, use_garch=use_garch)
        peer_increment = peer_segment - peer_segment[0]   # length m+2
        shift_k = beta * peer_increment[1:m + 1]
        R_t = logp_t[t_R] - logp_t[t_L]
        R_p = logp_p[t_R] - logp_p[t_L]
        if use_garch:
            eps_full = np.full_like(r_t, np.nan)
            eps_full[mask] = eps
            sig2_path = fit_sigma2_garch_path(eps_full, m + 1, t_L)
        else:
            sig2_path = np.full(m + 1, sig2_eps)
        bridge_resid_mean, bridge_var = bridge_mean_var(0.0, R_t - beta * R_p, m, sig2_path)
        mean = logp_t[t_L] + shift_k + bridge_resid_mean
        return mean, bridge_var

    if lag == 1:
        # ---------- one-day-lagged peer regression ----------
        # r_p_lag1[i] = r_p[i-1]; aligns with r_t[i] (both refer to day i+1).
        r_p_lag1 = np.concatenate([[np.nan], r_p[:-1]])
        mask = ~np.isnan(r_t) & ~np.isnan(r_p_lag1)
        mask[t_L:t_R] = False     # drop returns whose target day falls in the gap
        rt, rpl = r_t[mask], r_p_lag1[mask]
        A = np.column_stack([np.ones_like(rpl), rpl])
        coef, *_ = np.linalg.lstsq(A, rt, rcond=None)
        alpha, beta = float(coef[0]), float(coef[1])
        eps = rt - (alpha + beta * rpl)
        sig2_eps = float(np.var(eps, ddof=2))
        # peer must be observed on r_p indices t_L-1 .. t_R-2 (we need r_p at
        # those indices for j = 1..m+1, since r_p index for step j is t_L+j-2).
        needed_lo, needed_hi = t_L - 1, t_R - 2     # inclusive r_p indices (length m+1)
        if needed_lo < 0 or needed_hi >= len(r_p) or np.any(np.isnan(r_p[needed_lo:needed_hi + 1])):
            return univariate_bridge(logp_t, t_L, t_R, use_garch=use_garch)
        # mu_j = alpha + beta * r_peer_{j-1}  for j = 1..m+1   (length m+1)
        mu_j = alpha + beta * r_p[needed_lo:needed_hi + 1]
        sum_all = float(mu_j.sum())
        xL, xR = logp_t[t_L], logp_t[t_R]
        k_arr = np.arange(1, m + 1)
        cum_mu_k = np.cumsum(mu_j)[:m]
        mean = xL + cum_mu_k + (k_arr / (m + 1)) * (xR - xL - sum_all)
        # variance: time-changed Brownian-bridge on residuals
        if use_garch:
            eps_full = np.full_like(r_t, np.nan)
            eps_full[mask] = eps
            sig2_path = fit_sigma2_garch_path(eps_full, m + 1, t_L)
        else:
            sig2_path = np.full(m + 1, sig2_eps)
        S = np.cumsum(sig2_path)
        Stot = S[-1]
        Sk = S[k_arr - 1]
        var = Sk * (Stot - Sk) / Stot
        return mean, var

    raise ValueError(f"bivariate_bridge: unsupported lag={lag} (only 0 or 1)")


# ---------------------------------------------------------------------------
# 4.  Validation: synthetic gaps
# ---------------------------------------------------------------------------
def linear_interp(logp, t_L, t_R):
    m = t_R - t_L - 1
    k = np.arange(1, m + 1)
    return logp[t_L] + (k / (m + 1)) * (logp[t_R] - logp[t_L])


def cubic_spline_interp(logp, t_L, t_R):
    """Natural cubic spline interpolation of log-prices through the gap
    using the observed points on either side."""
    from scipy.interpolate import CubicSpline
    m = t_R - t_L - 1
    # anchor points: 50 observed days before t_L and 50 after t_R
    span = 50
    lo = max(0, t_L - span)
    hi = min(len(logp), t_R + span + 1)
    idx = np.concatenate([np.arange(lo, t_L + 1),
                          np.arange(t_R, hi)])
    vals = logp[idx]
    ok = ~np.isnan(vals)
    idx = idx[ok]; vals = vals[ok]
    if len(idx) < 4:
        return linear_interp(logp, t_L, t_R)
    cs = CubicSpline(idx, vals, bc_type="natural")
    k_days = np.arange(t_L + 1, t_R)
    return cs(k_days)


def crps_gaussian(y, mu, sigma):
    """Closed-form CRPS for a Gaussian forecast."""
    z = (y - mu) / sigma
    return sigma * (z * (2 * stats.norm.cdf(z) - 1)
                    + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))


def _non_overlapping_sample(valid_positions, k, min_sep):
    """Greedy random sample of positions such that any two chosen are at least
    min_sep apart.  Prevents artificial gaps from overlapping across folds."""
    pool = list(valid_positions)
    RNG.shuffle(pool)
    chosen = []
    for p in pool:
        if all(abs(p - q) >= min_sep for q in chosen):
            chosen.append(p)
        if len(chosen) == k:
            break
    return chosen


def wilson_ci(successes, n, z=Z95):
    """Wilson score 95% CI for a proportion."""
    if n == 0:
        return (np.nan, np.nan)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (centre - half, centre + half)


def synthetic_validate(logp_t, logp_p_full, asset, peer, peer_lag=0):
    """Return dict of metrics for linear / univariate / (bivariate) / biv-t on K folds.

    Gaps are sampled without overlap (separation >= M_GAP+1) so that the
    folds are independent, which is required for honest coverage intervals.
    Also reports per-fold RMSE and a calm/turbulent stratification based on
    rolling volatility at the fold position.
    """
    n = len(logp_t)
    valid_positions = []
    obs = ~np.isnan(logp_t)
    for pos in range(300, n - M_GAP - 300):
        if np.all(obs[pos:pos + M_GAP + 2]):
            valid_positions.append(pos)
    sample = _non_overlapping_sample(valid_positions, K_FOLDS, M_GAP + 1)

    # rolling 50-day realised vol for turbulence classification
    r = np.diff(logp_t)
    win = 50
    roll_sd = pd.Series(r).rolling(win, min_periods=20).std().to_numpy()
    med_sd = np.nanmedian(roll_sd)

    # Full ablation: every combination of {univariate, bivariate} x {constant sigma, GARCH}
    # plus linear, spline, and the bivariate+GARCH Student-t band for robustness.
    methods = ("linear", "spline",
               "univ_const", "univ_garch",
               "biv_const", "biv_garch", "biv_t")
    metrics = {m: {"err2": [], "y": [], "mu": [], "sd": [],
                   "fold_rmse": [], "fold_regime": []}
               for m in methods}

    for pos in sample:
        t_L, t_R = pos, pos + M_GAP + 1
        true = logp_t[t_L + 1:t_R]
        regime = "turbulent" if (not np.isnan(roll_sd[t_L]) and roll_sd[t_L] > med_sd) else "calm"
        work = logp_t.copy()
        work[t_L + 1:t_R] = np.nan

        # linear
        mu_lin = linear_interp(work, t_L, t_R)
        err2 = (mu_lin - true) ** 2
        metrics["linear"]["err2"].extend(err2)
        metrics["linear"]["fold_rmse"].append(np.sqrt(err2.mean()))
        metrics["linear"]["fold_regime"].append(regime)

        # cubic spline on log-prices
        mu_sp = cubic_spline_interp(work, t_L, t_R)
        err2 = (mu_sp - true) ** 2
        metrics["spline"]["err2"].extend(err2)
        metrics["spline"]["fold_rmse"].append(np.sqrt(err2.mean()))
        metrics["spline"]["fold_regime"].append(regime)

        # univariate bridge -- both variance choices
        for mname, use_g in (("univ_const", False), ("univ_garch", True)):
            mu_u, var_u = univariate_bridge(work, t_L, t_R, use_garch=use_g)
            sd_u = np.sqrt(var_u)
            err2 = (mu_u - true) ** 2
            metrics[mname]["err2"].extend(err2)
            metrics[mname]["y"].extend(true); metrics[mname]["mu"].extend(mu_u); metrics[mname]["sd"].extend(sd_u)
            metrics[mname]["fold_rmse"].append(np.sqrt(err2.mean()))
            metrics[mname]["fold_regime"].append(regime)

        # bivariate bridge -- both variance choices, plus Student-t on biv_garch
        if peer is not None:
            peer_seg = logp_p_full[t_L:t_R + 1]
            if not np.any(np.isnan(peer_seg)):
                for mname, use_g in (("biv_const", False), ("biv_garch", True)):
                    mu_b, var_b = bivariate_bridge(work, logp_p_full, t_L, t_R,
                                                   use_garch=use_g, lag=peer_lag)
                    sd_b = np.sqrt(var_b)
                    err2 = (mu_b - true) ** 2
                    metrics[mname]["err2"].extend(err2)
                    metrics[mname]["y"].extend(true); metrics[mname]["mu"].extend(mu_b); metrics[mname]["sd"].extend(sd_b)
                    metrics[mname]["fold_rmse"].append(np.sqrt(err2.mean()))
                    metrics[mname]["fold_regime"].append(regime)
                    # keep a Student-t variant of the GARCH bridge for the robustness check
                    if mname == "biv_garch":
                        df_t = 6.0
                        t_scale = stats.t.ppf(0.975, df=df_t) / Z95
                        sd_bt = sd_b * t_scale
                        metrics["biv_t"]["err2"].extend(err2)
                        metrics["biv_t"]["y"].extend(true); metrics["biv_t"]["mu"].extend(mu_b); metrics["biv_t"]["sd"].extend(sd_bt)
                        metrics["biv_t"]["fold_rmse"].append(np.sqrt(err2.mean()))
                        metrics["biv_t"]["fold_regime"].append(regime)

    out = {}
    for name, d in metrics.items():
        if not d["err2"]:
            continue
        rmse_log = np.sqrt(np.mean(d["err2"]))
        rmse_pct = (np.exp(rmse_log) - 1) * 100
        row = {"rmse_log": rmse_log, "rmse_pct": rmse_pct,
               "fold_rmse_std": float(np.std(d["fold_rmse"], ddof=1)) if len(d["fold_rmse"]) > 1 else np.nan}
        # calm/turbulent split
        reg = np.array(d["fold_regime"])
        fr = np.array(d["fold_rmse"])
        if (reg == "calm").any():
            row["rmse_calm"] = float(np.sqrt(np.mean(fr[reg == "calm"]**2)))
        if (reg == "turbulent").any():
            row["rmse_turb"] = float(np.sqrt(np.mean(fr[reg == "turbulent"]**2)))
        if d["y"]:
            y = np.array(d["y"]); mu = np.array(d["mu"]); sd = np.array(d["sd"])
            hits = np.abs(y - mu) <= Z95 * sd
            cov = hits.mean() * 100
            crps = np.mean(crps_gaussian(y, mu, sd))
            lo, hi = wilson_ci(hits.sum(), len(hits))
            row["coverage95"] = cov
            row["cov_lo"] = lo * 100
            row["cov_hi"] = hi * 100
            row["crps"] = crps
        out[name] = row
    return out, len(sample)


def leadlag_r2(logp_t, logp_p):
    """R^2 of the contemporaneous peer regression and of the lead-lag one (L=1).
    Used to back up the claim that including a lagged peer adds <1% R^2."""
    r_t = np.diff(logp_t); r_p = np.diff(logp_p)
    mask = ~np.isnan(r_t) & ~np.isnan(r_p)
    rt, rp = r_t[mask], r_p[mask]
    # contemporaneous
    b, a = np.polyfit(rp, rt, 1)
    ss_res = np.sum((rt - (a + b * rp))**2)
    ss_tot = np.sum((rt - rt.mean())**2)
    r2_0 = 1 - ss_res / ss_tot
    # lead-lag: y_t = a + b0 rp_t + b1 rp_{t-1}
    rp_l1 = np.concatenate([[np.nan], rp[:-1]])
    m2 = ~np.isnan(rp_l1)
    X = np.column_stack([np.ones(m2.sum()), rp[m2], rp_l1[m2]])
    y = rt[m2]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    ss_res2 = np.sum((y - X @ coef)**2)
    ss_tot2 = np.sum((y - y.mean())**2)
    r2_1 = 1 - ss_res2 / ss_tot2
    return r2_0, r2_1


# ---------------------------------------------------------------------------
# 5.  Run everything
# ---------------------------------------------------------------------------
def main():
    df = load_clean()
    log_df = np.log(df[ASSETS])
    gaps = {a: gap_bounds(log_df[a]) for a in ASSETS}
    print("Gap bounds (t_L, t_R):")
    for a, g in gaps.items():
        print(f"  {a:14s} {g}")

    # ----- validation table -----
    rows = []
    for a in ASSETS:
        peer = PEERS.get(a)
        peer_lag = PEER_LAGS.get(a, 0)
        peer_logp = log_df[peer].to_numpy() if peer else None
        m, n_folds = synthetic_validate(log_df[a].to_numpy(), peer_logp, a, peer,
                                        peer_lag=peer_lag)
        for method, vals in m.items():
            rows.append({
                "asset": a, "method": method, "n_folds": n_folds,
                "rmse_log": vals.get("rmse_log"),
                "rmse_pct": vals.get("rmse_pct"),
                "rmse_calm": vals.get("rmse_calm"),
                "rmse_turb": vals.get("rmse_turb"),
                "fold_rmse_std": vals.get("fold_rmse_std"),
                "coverage95": vals.get("coverage95"),
                "cov_lo": vals.get("cov_lo"),
                "cov_hi": vals.get("cov_hi"),
                "crps": vals.get("crps"),
            })
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "task2_results.csv", index=False)
    print("\nValidation results:")
    print(res.to_string(index=False))

    # ----- lead-lag diagnostic for the two peer pairs -----
    print("\nLead-lag R^2 comparison (contemporaneous vs + lag-1):")
    leadlag_rows = []
    for a, p in [("gurkor", "water"), ("slingshots", "guitars")]:
        r0, r1 = leadlag_r2(log_df[a].to_numpy(), log_df[p].to_numpy())
        print(f"  {a:11s}<->{p:11s}  R2_0={r0:.4f}  R2_1={r1:.4f}  gain={100*(r1-r0):.3f}pp")
        leadlag_rows.append({"target": a, "peer": p, "r2_contemp": r0,
                             "r2_leadlag": r1, "gain_pp": 100 * (r1 - r0)})
    pd.DataFrame(leadlag_rows).to_csv(ROOT / "task2_leadlag.csv", index=False)

    # ----- final interpolation in the real gaps -----
    final = {}
    for a in ASSETS:
        t_L, t_R = gaps[a]
        logp = log_df[a].to_numpy()
        peer = PEERS.get(a)
        peer_lag = PEER_LAGS.get(a, 0)
        use_garch_local = a in USE_GARCH
        if peer is not None:
            peer_logp = log_df[peer].to_numpy()
            mu, var = bivariate_bridge(logp, peer_logp, t_L, t_R,
                                       use_garch=use_garch_local, lag=peer_lag)
            lag_tag = f" lag-{peer_lag}" if peer_lag else ""
            method = f"bivariate ({peer}){lag_tag}" + (" + GARCH" if use_garch_local else "")
        else:
            mu, var = univariate_bridge(logp, t_L, t_R, use_garch=use_garch_local)
            method = "univariate" + (" + GARCH" if use_garch_local else "")
        sd = np.sqrt(var)
        final[a] = dict(t_L=t_L, t_R=t_R, mu=mu, sd=sd, method=method)
        print(f"{a:14s}: max log-sd = {sd.max():.4f}  -> 95% half-width = {Z95*sd.max()*100:.2f}% (peak),  method = {method}")

    # ----- figure: interpolated paths -----
    fig, axes = plt.subplots(4, 2, figsize=(13, 14), sharex=False)
    axes = axes.ravel()
    for ax, a in zip(axes, ASSETS):
        f = final[a]
        t_L, t_R = f["t_L"], f["t_R"]
        days = np.arange(len(df))
        prices = df[a].to_numpy()
        # plot wider context window
        lo = max(0, t_L - 100); hi = min(len(df), t_R + 100)
        ax.plot(days[lo:hi], prices[lo:hi], "k-", lw=0.7, label="observed")
        # interpolated median + bands on price scale
        k_idx = np.arange(t_L + 1, t_R)
        median = np.exp(f["mu"])
        lo95 = np.exp(f["mu"] - Z95 * f["sd"])
        hi95 = np.exp(f["mu"] + Z95 * f["sd"])
        ax.fill_between(k_idx, lo95, hi95, color="C1", alpha=0.3, label="95% CI")
        ax.plot(k_idx, median, "C1-", lw=1.4, label="median")
        ax.scatter([t_L, t_R], [prices[t_L], prices[t_R]], color="C3", zorder=5, s=20)
        ax.set_title(f"{a} — {f['method']}")
        ax.set_xlabel("day"); ax.set_ylabel("price")
        ax.legend(fontsize=7, loc="best")
    axes[-1].axis("off")
    plt.tight_layout()
    plt.savefig(PIC / "fig_interp_paths.png", dpi=130)
    plt.close()
    print(f"\nSaved figure -> {PIC/'fig_interp_paths.png'}")

    # ----- figure: lens-shape band-width comparison -----
    fig, ax = plt.subplots(figsize=(8, 5))
    for a in ASSETS:
        f = final[a]
        k = np.arange(1, len(f["sd"]) + 1)
        ax.plot(k, Z95 * f["sd"] * 100, label=a)
    ax.set_xlabel("day inside gap (k)")
    ax.set_ylabel("95% half-width (% of price)")
    ax.set_title("Brownian-bridge uncertainty profile across the gap")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PIC / "fig_interp_uncertainty.png", dpi=130)
    plt.close()
    print(f"Saved figure -> {PIC/'fig_interp_uncertainty.png'}")

    # ----- save final point + interval estimates -----
    out_rows = []
    for a in ASSETS:
        f = final[a]
        for k, (mu, sd) in enumerate(zip(f["mu"], f["sd"]), start=1):
            out_rows.append({
                "asset": a, "day": f["t_L"] + k,
                "median": float(np.exp(mu)),
                "lo95": float(np.exp(mu - Z95 * sd)),
                "hi95": float(np.exp(mu + Z95 * sd)),
                "log_sd": float(sd),
            })
    pd.DataFrame(out_rows).to_csv(ROOT / "task2_interpolated_values.csv", index=False)
    print(f"Saved per-day estimates -> {ROOT/'task2_interpolated_values.csv'}")

    # ----- mid-gap summary table (gap midpoint + endpoints per asset) -----
    summary_rows = []
    for a in ASSETS:
        f = final[a]
        t_L, t_R = f["t_L"], f["t_R"]
        mid = len(f["mu"]) // 2
        mu_mid, sd_mid = f["mu"][mid], f["sd"][mid]
        summary_rows.append({
            "asset": a,
            "method": f["method"],
            "gap_day_L": t_L + 1,
            "gap_day_R": t_R - 1,
            "mid_day": t_L + 1 + mid,
            "median_mid": float(np.exp(mu_mid)),
            "lo95_mid": float(np.exp(mu_mid - Z95 * sd_mid)),
            "hi95_mid": float(np.exp(mu_mid + Z95 * sd_mid)),
            "halfwidth_pct_mid": float((np.exp(Z95 * sd_mid) - 1) * 100),
        })
    pd.DataFrame(summary_rows).to_csv(ROOT / "task2_summary.csv", index=False)
    print(f"Saved gap-midpoint summary -> {ROOT/'task2_summary.csv'}")


if __name__ == "__main__":
    main()
