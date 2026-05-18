"""
Task 2 — Six additional rows for the synthetic-gap validation table.

Imports the production module (`task2_interpolation`) so the fold sampler,
regime classifier, Wilson CI, CRPS, and outlier rule are all reused exactly.
This script does NOT modify the production code.

Adds:
  (A) Sugar with slingshots as peer, LAG-1 bivariate bridge (const sigma + GARCH).
  (B) Tranquillity with its empirical best lag-0 peer (const + GARCH).
  (C) Stocks with its empirical best lag-0 peer (const + GARCH).

Also runs `univ_const` on sugar/tranq/stocks through the same wrapper as a
sanity check that the fold sample is identical to the production table.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy import stats
from arch import arch_model

# import production helpers (read-only)
PROD = Path("/sessions/charming-kind-dirac/mnt/TMS088 Project Code/Task 2")
sys.path.insert(0, str(PROD))
import task2_interpolation as t2  # noqa: E402

ASSETS = t2.ASSETS
M_GAP = t2.M_GAP
K_FOLDS = t2.K_FOLDS
Z95 = t2.Z95


# ---------------------------------------------------------------------------
# LAG-1 bivariate bridge (Sugar / Slingshots)
# ---------------------------------------------------------------------------
def biv_bridge_lag1(logp_t, logp_p, t_L, t_R, use_garch=False):
    """
    Bivariate bridge using LAG-1 peer return as covariate:
        r_t[s] = alpha + beta * r_p[s-1] + eps[s]
    Inside the gap the conditional mean drift mu_j = alpha + beta * r_p[t_L+j-1]
    is summed step by step.  The bridge correction distributes the residual
    (x_R - x_L) - sum_{j=1..m+1} mu_j across days (proportional to accumulated
    variance, matching the production bridge_mean_var; for constant sigma this
    is linear in k).
    Returns (mean[1..m], var[1..m]).
    """
    m = t_R - t_L - 1
    r_t = np.diff(logp_t)          # r_t[s] = logp[s+1] - logp[s], length n-1
    r_p = np.diff(logp_p)

    # in-sample regression: r_t[s] on r_p[s-1] for s outside the gap window
    n_r = len(r_t)
    valid = np.ones(n_r, dtype=bool)
    valid &= ~np.isnan(r_t)
    r_p_l1 = np.full(n_r, np.nan)
    r_p_l1[1:] = r_p[:-1]
    valid &= ~np.isnan(r_p_l1)
    s_idx = np.arange(n_r)
    valid &= ~((s_idx >= t_L) & (s_idx <= t_R - 1))

    y = r_t[valid]
    x = r_p_l1[valid]
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    alpha, beta = float(coef[0]), float(coef[1])
    resid = y - (alpha + beta * x)
    sig2_eps = float(np.var(resid, ddof=2))

    # in-gap step drifts mu_j for j=1..m+1
    # step j corresponds to return r_t[t_L+j-1], which uses r_p[t_L+j-2]
    mu_steps = np.empty(m + 1)
    for j in range(1, m + 2):
        lag_idx = (t_L + j - 1) - 1
        if lag_idx < 0 or lag_idx >= n_r or np.isnan(r_p[lag_idx]):
            mu_steps[j - 1] = alpha
        else:
            mu_steps[j - 1] = alpha + beta * r_p[lag_idx]

    # variance path
    if use_garch:
        resid_full = np.full(n_r, np.nan)
        resid_full[valid] = y - (alpha + beta * x)
        seg = resid_full[:t_L]
        seg = seg[~np.isnan(seg)]
        if len(seg) > 1000:
            seg = seg[-1000:]
        if len(seg) < 200:
            sig2_path = np.full(m + 1, sig2_eps)
        else:
            try:
                am = arch_model(seg * 100, vol="Garch", p=1, q=1,
                                mean="Zero", rescale=False, dist="normal")
                res = am.fit(disp="off", show_warning=False)
                params = res.params
                omega = float(params["omega"]) / 1e4
                a1 = float(params["alpha[1]"])
                b1 = float(params["beta[1]"])
                cv = np.asarray(res.conditional_volatility)
                v0 = float(cv[-1] ** 2) / 1e4
                v = np.empty(m + 1)
                v_prev = v0
                for j in range(m + 1):
                    v_prev = omega + (a1 + b1) * v_prev
                    v[j] = v_prev
                sig2_path = v
            except Exception:
                sig2_path = np.full(m + 1, sig2_eps)
    else:
        sig2_path = np.full(m + 1, sig2_eps)

    # bridge correction (proportional to accumulated variance)
    pre_cum = np.cumsum(mu_steps)
    total_mu = pre_cum[-1]
    R = (logp_t[t_R] - logp_t[t_L]) - total_mu
    S = np.cumsum(sig2_path)
    Stot = S[-1]
    Sk = S[:m]
    bridge_var = Sk * (Stot - Sk) / Stot
    bridge_mean_corr = (Sk / Stot) * R
    pre_mean = pre_cum[:m]
    mean = logp_t[t_L] + pre_mean + bridge_mean_corr
    return mean, bridge_var


# ---------------------------------------------------------------------------
# Wrapper that replicates synthetic_validate but only for chosen methods,
# using the SAME fold sample as the production code.
# ---------------------------------------------------------------------------
def _replay_rng_until(asset, log_df):
    """
    The production main() loops through ASSETS in order and each call to
    synthetic_validate consumes exactly ONE RNG.shuffle (inside
    _non_overlapping_sample).  We replay those shuffles so our sample
    matches the production fold positions for the given asset.
    """
    t2.RNG = np.random.default_rng(20260414)
    for a in ASSETS:
        if a == asset:
            return
        logp = log_df[a].to_numpy()
        obs = ~np.isnan(logp)
        n = len(logp)
        valid_positions = []
        for pos in range(300, n - M_GAP - 300):
            if np.all(obs[pos:pos + M_GAP + 2]):
                valid_positions.append(pos)
        # consume one shuffle by calling the sampler
        t2._non_overlapping_sample(valid_positions, K_FOLDS, M_GAP + 1)


def validate_methods(asset, logp_t, peer_logp, methods_spec, log_df):
    # replay RNG state so our sample matches production for `asset`
    _replay_rng_until(asset, log_df)

    n = len(logp_t)
    obs = ~np.isnan(logp_t)
    valid_positions = []
    for pos in range(300, n - M_GAP - 300):
        if np.all(obs[pos:pos + M_GAP + 2]):
            valid_positions.append(pos)
    sample = t2._non_overlapping_sample(valid_positions, K_FOLDS, M_GAP + 1)

    r = np.diff(logp_t)
    win = 50
    roll_sd = pd.Series(r).rolling(win, min_periods=20).std().to_numpy()
    med_sd = np.nanmedian(roll_sd)

    metrics = {spec["name"]: {"err2": [], "y": [], "mu": [], "sd": [],
                              "fold_rmse": [], "fold_regime": []}
               for spec in methods_spec}

    for pos in sample:
        t_L, t_R = pos, pos + M_GAP + 1
        true = logp_t[t_L + 1:t_R]
        regime = ("turbulent"
                  if (not np.isnan(roll_sd[t_L]) and roll_sd[t_L] > med_sd)
                  else "calm")
        work = logp_t.copy()
        work[t_L + 1:t_R] = np.nan

        for spec in methods_spec:
            name = spec["name"]
            kind = spec["kind"]
            if kind == "univ_const":
                mu, var = t2.univariate_bridge(work, t_L, t_R, use_garch=False)
            elif kind == "biv_lag0_const":
                mu, var = t2.bivariate_bridge(work, peer_logp, t_L, t_R,
                                              use_garch=False)
            elif kind == "biv_lag0_garch":
                mu, var = t2.bivariate_bridge(work, peer_logp, t_L, t_R,
                                              use_garch=True)
            elif kind == "biv_lag1_const":
                mu, var = biv_bridge_lag1(work, peer_logp, t_L, t_R,
                                          use_garch=False)
            elif kind == "biv_lag1_garch":
                mu, var = biv_bridge_lag1(work, peer_logp, t_L, t_R,
                                          use_garch=True)
            else:
                raise ValueError(kind)
            sd = np.sqrt(var)
            err2 = (mu - true) ** 2
            metrics[name]["err2"].extend(err2)
            metrics[name]["y"].extend(true)
            metrics[name]["mu"].extend(mu)
            metrics[name]["sd"].extend(sd)
            metrics[name]["fold_rmse"].append(np.sqrt(err2.mean()))
            metrics[name]["fold_regime"].append(regime)

    out = {}
    for name, d in metrics.items():
        rmse_log = np.sqrt(np.mean(d["err2"]))
        rmse_pct = (np.exp(rmse_log) - 1) * 100
        reg = np.array(d["fold_regime"])
        fr = np.array(d["fold_rmse"])
        row = {"rmse_log": rmse_log, "rmse_pct": rmse_pct}
        row["rmse_calm_log"] = (float(np.sqrt(np.mean(fr[reg == "calm"] ** 2)))
                                if (reg == "calm").any() else np.nan)
        row["rmse_turb_log"] = (float(np.sqrt(np.mean(fr[reg == "turbulent"] ** 2)))
                                if (reg == "turbulent").any() else np.nan)
        y = np.array(d["y"])
        mu = np.array(d["mu"])
        sd = np.array(d["sd"])
        hits = np.abs(y - mu) <= Z95 * sd
        cov = hits.mean() * 100
        crps = float(np.mean(t2.crps_gaussian(y, mu, sd)))
        lo, hi = t2.wilson_ci(hits.sum(), len(hits))
        row["coverage95"] = cov
        row["cov_lo"] = lo * 100
        row["cov_hi"] = hi * 100
        row["crps"] = crps
        row["n_calm"] = int((reg == "calm").sum())
        row["n_turb"] = int((reg == "turbulent").sum())
        out[name] = row
    return out, len(sample)


# ---------------------------------------------------------------------------
# Empirical peer selection (lag-0 max |rho|)
# ---------------------------------------------------------------------------
def best_lag0_peer(target_logp, candidates_logp_dict):
    r_t = np.diff(target_logp)
    out = []
    for name, peer_logp in candidates_logp_dict.items():
        r_p = np.diff(peer_logp)
        mask = ~np.isnan(r_t) & ~np.isnan(r_p)
        if mask.sum() < 30:
            continue
        rho = float(np.corrcoef(r_t[mask], r_p[mask])[0, 1])
        out.append((name, rho))
    out.sort(key=lambda x: abs(x[1]), reverse=True)
    return out


def to_pct_log(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    return (np.exp(x) - 1) * 100


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df = t2.load_clean()
    log_df = np.log(df[ASSETS])

    # ---- pick empirical peers ----
    print("=" * 70)
    print("Empirical peer selection (lag-0, |rho| of log-return correlation)")
    print("=" * 70)
    chosen = {}
    for tgt in ("tranquillity", "stocks"):
        cand = {a: log_df[a].to_numpy() for a in ASSETS if a != tgt}
        rank = best_lag0_peer(log_df[tgt].to_numpy(), cand)
        top, runner = rank[0], rank[1]
        chosen[tgt] = top[0]
        print(f"  {tgt:13s} -> best peer = {top[0]:12s} |rho|={abs(top[1]):.4f} "
              f"(rho={top[1]:+.4f}); runner-up = {runner[0]:12s} "
              f"|rho|={abs(runner[1]):.4f}")

    # ---- validations ----
    print("\n" + "=" * 70)
    print("Running validations (production fold sampler)")
    print("=" * 70)

    sugar_lp = log_df["sugar"].to_numpy()
    slings_lp = log_df["slingshots"].to_numpy()
    tranq_lp = log_df["tranquillity"].to_numpy()
    stocks_lp = log_df["stocks"].to_numpy()

    sugar_specs = [
        {"name": "univ_const_sanity", "kind": "univ_const"},
        {"name": "biv_slings_lag1_const", "kind": "biv_lag1_const"},
        {"name": "biv_slings_lag1_garch", "kind": "biv_lag1_garch"},
    ]
    sugar_res, sugar_nf = validate_methods("sugar", sugar_lp, slings_lp,
                                           sugar_specs, log_df)

    tranq_peer_lp = log_df[chosen["tranquillity"]].to_numpy()
    tranq_specs = [
        {"name": "univ_const_sanity", "kind": "univ_const"},
        {"name": "biv_tranq_const", "kind": "biv_lag0_const"},
        {"name": "biv_tranq_garch", "kind": "biv_lag0_garch"},
    ]
    tranq_res, tranq_nf = validate_methods("tranquillity", tranq_lp,
                                           tranq_peer_lp, tranq_specs, log_df)

    stocks_peer_lp = log_df[chosen["stocks"]].to_numpy()
    stocks_specs = [
        {"name": "univ_const_sanity", "kind": "univ_const"},
        {"name": "biv_stocks_const", "kind": "biv_lag0_const"},
        {"name": "biv_stocks_garch", "kind": "biv_lag0_garch"},
    ]
    stocks_res, stocks_nf = validate_methods("stocks", stocks_lp, stocks_peer_lp,
                                             stocks_specs, log_df)

    # ---- sanity check ----
    print("\nSanity check (univ_const rmse_pct vs production table):")
    expect = {"sugar": 4.20, "tranquillity": 3.55, "stocks": 4.42}
    for asset, res, nf in (("sugar", sugar_res, sugar_nf),
                           ("tranquillity", tranq_res, tranq_nf),
                           ("stocks", stocks_res, stocks_nf)):
        rmse_pct = res["univ_const_sanity"]["rmse_pct"]
        ok = abs(rmse_pct - expect[asset]) < 0.05
        print(f"  {asset:13s} n_folds={nf:3d} "
              f"univ_const rmse_pct={rmse_pct:.4f} (expected ~{expect[asset]}) "
              f"{'OK' if ok else 'MISMATCH'}")

    # ---- CSV block ----
    print("\n" + "=" * 70)
    print("CSV results (new methods)")
    print("=" * 70)
    print("asset,method,n_folds,rmse_pct,rmse_calm,rmse_turb,coverage95,"
          "cov_lo,cov_hi,crps")
    blocks = [
        ("sugar", sugar_nf, [
            ("biv_slings_lag1_const", sugar_res["biv_slings_lag1_const"]),
            ("biv_slings_lag1_garch", sugar_res["biv_slings_lag1_garch"]),
        ]),
        ("tranquillity", tranq_nf, [
            ("biv_tranq_const", tranq_res["biv_tranq_const"]),
            ("biv_tranq_garch", tranq_res["biv_tranq_garch"]),
        ]),
        ("stocks", stocks_nf, [
            ("biv_stocks_const", stocks_res["biv_stocks_const"]),
            ("biv_stocks_garch", stocks_res["biv_stocks_garch"]),
        ]),
    ]
    for asset, nf, items in blocks:
        for name, r in items:
            rmse_calm_pct = to_pct_log(r["rmse_calm_log"])
            rmse_turb_pct = to_pct_log(r["rmse_turb_log"])
            print(f"{asset},{name},{nf},{r['rmse_pct']:.4f},"
                  f"{rmse_calm_pct:.4f},{rmse_turb_pct:.4f},"
                  f"{r['coverage95']:.4f},{r['cov_lo']:.4f},"
                  f"{r['cov_hi']:.4f},{r['crps']:.6f}")

    # ---- LaTeX ----
    print("\n" + "=" * 70)
    print("LaTeX rows")
    print("=" * 70)

    tranq_peer = chosen["tranquillity"]
    stocks_peer = chosen["stocks"]

    def fmt_row(label, r):
        rmse_calm_pct = to_pct_log(r["rmse_calm_log"])
        rmse_turb_pct = to_pct_log(r["rmse_turb_log"])
        return (f"& {label:<42s} & {r['rmse_pct']:.2f} & "
                f"{rmse_calm_pct:.2f} & {rmse_turb_pct:.2f} & "
                f"{r['coverage95']:.1f} & "
                f"[{r['cov_lo']:.1f}, {r['cov_hi']:.1f}] & "
                f"{r['crps']:.5f} \\\\")

    print(fmt_row("biv. (slingshots) lag-1, const $\\sigma$",
                  sugar_res["biv_slings_lag1_const"]))
    print(fmt_row("biv. (slingshots) lag-1 + GARCH",
                  sugar_res["biv_slings_lag1_garch"]))
    print(fmt_row(f"biv. ({tranq_peer}), const $\\sigma$",
                  tranq_res["biv_tranq_const"]))
    print(fmt_row(f"biv. ({tranq_peer}) + GARCH",
                  tranq_res["biv_tranq_garch"]))
    print(fmt_row(f"biv. ({stocks_peer}), const $\\sigma$",
                  stocks_res["biv_stocks_const"]))
    print(fmt_row(f"biv. ({stocks_peer}) + GARCH",
                  stocks_res["biv_stocks_garch"]))


if __name__ == "__main__":
    main()
