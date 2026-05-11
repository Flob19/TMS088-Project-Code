"""
Extra figures for Task 2 (per Gemini review):
  fig_interp_compare.png    — linear vs bridge, for gurkor and stocks
  fig_garch_variance.png    — GARCH(1,1) variance path inside the gap
  fig_synth_fold_example.png — one synthetic fold with truth + estimate + band
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from task2_interpolation import (
    load_clean, gap_bounds, linear_interp,
    univariate_bridge, bivariate_bridge,
    fit_sigma2_garch_path,
    ASSETS, PEERS, USE_GARCH, M_GAP, Z95, PIC, RNG,
)


def main():
    df = load_clean()
    log_df = np.log(df[ASSETS])
    gaps = {a: gap_bounds(log_df[a]) for a in ASSETS}

    # ---------------------------------------------------------------
    # 1. Linear vs bridge for one bivariate-peer asset and one solo
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, a in zip(axes, ["gurkor", "stocks"]):
        t_L, t_R = gaps[a]
        logp = log_df[a].to_numpy()
        peer = PEERS.get(a)
        use_garch_local = a in USE_GARCH
        if peer is not None:
            mu_b, var_b = bivariate_bridge(logp, log_df[peer].to_numpy(),
                                           t_L, t_R, use_garch=use_garch_local)
            lbl_b = f"bivariate bridge ({peer})"
        else:
            mu_b, var_b = univariate_bridge(logp, t_L, t_R, use_garch=use_garch_local)
            lbl_b = "univariate bridge"
        sd_b = np.sqrt(var_b)
        mu_lin = linear_interp(logp, t_L, t_R)

        k_idx = np.arange(t_L + 1, t_R)
        prices = df[a].to_numpy()
        lo, hi = max(0, t_L - 60), min(len(df), t_R + 60)
        ax.plot(np.arange(lo, hi), prices[lo:hi], "k-", lw=0.7, label="observed")
        ax.fill_between(k_idx, np.exp(mu_b - Z95 * sd_b),
                        np.exp(mu_b + Z95 * sd_b),
                        color="C1", alpha=0.3, label="bridge 95% CI")
        ax.plot(k_idx, np.exp(mu_b), "C1-", lw=1.6, label=lbl_b)
        ax.plot(k_idx, np.exp(mu_lin), "C0--", lw=1.4, label="linear interpolation")
        ax.scatter([t_L, t_R], [prices[t_L], prices[t_R]],
                   color="C3", zorder=5, s=25, label="observed endpoints")
        ax.set_title(a)
        ax.set_xlabel("day"); ax.set_ylabel("price")
        ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(PIC / "fig_interp_compare.png", dpi=130)
    plt.close()
    print(f"Saved -> {PIC/'fig_interp_compare.png'}")

    # ---------------------------------------------------------------
    # 1b. Target (gurkor) + peer (water) side-by-side during the gap
    #     -- shows WHY the bivariate bridge's orange line is wavy
    # ---------------------------------------------------------------
    a, peer = "gurkor", "water"
    t_L, t_R = gaps[a]
    logp_t = log_df[a].to_numpy()
    logp_p = log_df[peer].to_numpy()
    mu_b, var_b = bivariate_bridge(logp_t, logp_p, t_L, t_R, use_garch=False)
    sd_b = np.sqrt(var_b)
    mu_lin = linear_interp(logp_t, t_L, t_R)
    k_idx = np.arange(t_L + 1, t_R)

    prices_t = df[a].to_numpy()
    prices_p = df[peer].to_numpy()
    lo, hi = max(0, t_L - 50), min(len(df), t_R + 50)
    span = np.arange(lo, hi)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True)

    # LEFT: gurkor (target) with gap, linear vs bridge
    axL.plot(span, prices_t[lo:hi], "k-", lw=0.7, label="observed")
    axL.fill_between(k_idx, np.exp(mu_b - Z95 * sd_b),
                     np.exp(mu_b + Z95 * sd_b),
                     color="C1", alpha=0.3, label="bridge 95% CI")
    axL.plot(k_idx, np.exp(mu_b), "C1-", lw=1.6, label="bivariate bridge (water)")
    axL.plot(k_idx, np.exp(mu_lin), "C0--", lw=1.4, label="linear interpolation")
    axL.scatter([t_L, t_R], [prices_t[t_L], prices_t[t_R]],
                color="C3", zorder=5, s=25, label="observed endpoints")
    axL.axvspan(t_L, t_R, color="grey", alpha=0.08)
    axL.set_title(f"Target: {a}  (gap on days {t_L+1}--{t_R-1})")
    axL.set_xlabel("day"); axL.set_ylabel("price")
    axL.legend(fontsize=8, loc="best")

    # RIGHT: water (peer) -- fully observed over the same window
    axR.plot(span, prices_p[lo:hi], "k-", lw=0.7, label="water (fully observed)")
    # highlight water's path INSIDE gurkor's gap
    axR.plot(np.arange(t_L, t_R + 1), prices_p[t_L:t_R + 1],
             color="C2", lw=2.2, label="water inside gurkor's gap")
    axR.scatter([t_L, t_R], [prices_p[t_L], prices_p[t_R]],
                color="C3", zorder=5, s=25)
    axR.axvspan(t_L, t_R, color="grey", alpha=0.08)
    axR.set_title(f"Peer: {peer}  (observed during {a}'s gap)")
    axR.set_xlabel("day"); axR.set_ylabel("price")
    axR.legend(fontsize=8, loc="best")

    plt.tight_layout()
    plt.savefig(PIC / "fig_target_vs_peer.png", dpi=130)
    plt.close()
    print(f"Saved -> {PIC/'fig_target_vs_peer.png'}")

    # ---------------------------------------------------------------
    # 2. GARCH variance path inside the gap for the three GARCH series
    # ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    for a in ["slingshots", "guitars", "sugar"]:
        t_L, t_R = gaps[a]
        m = t_R - t_L - 1
        logp = log_df[a].to_numpy()
        r = np.diff(logp)
        sig2_garch = fit_sigma2_garch_path(r, m, t_L)
        k = np.arange(1, m + 2)
        ax.plot(k, np.sqrt(sig2_garch) * 100, label=f"{a}: GARCH $\\hat\\sigma_t$")
        # unconditional level (constant pre-gap sigma) for reference
        lo = max(0, t_L - 250)
        sigma_uncond = np.sqrt(np.nanvar(r[lo:t_L], ddof=1))
        ax.axhline(sigma_uncond * 100, ls="--", color=ax.lines[-1].get_color(),
                   alpha=0.5, lw=0.8)
    ax.set_xlabel("day inside gap (k)")
    ax.set_ylabel("daily volatility $\\hat\\sigma_t$ (% of log-price)")
    ax.set_title("GARCH(1,1) variance projection through the 50-day gap\n"
                 "dashed lines = unconditional pre-gap $\\sigma$")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PIC / "fig_garch_variance.png", dpi=130)
    plt.close()
    print(f"Saved -> {PIC/'fig_garch_variance.png'}")

    # ---------------------------------------------------------------
    # 3. Example synthetic fold: truth + bridge estimate + band
    # ---------------------------------------------------------------
    a = "slingshots"           # high-volatility so band is visible
    peer = PEERS[a]
    logp_t = log_df[a].to_numpy()
    logp_p = log_df[peer].to_numpy()
    n = len(logp_t)
    # find a turbulent-looking position in the middle of the sample
    obs_t = ~np.isnan(logp_t); obs_p = ~np.isnan(logp_p)
    valid = [p for p in range(1500, n - M_GAP - 500)
             if obs_t[p:p + M_GAP + 2].all() and obs_p[p:p + M_GAP + 2].all()]
    # pick a mid-high-vol fold deterministically for reproducibility
    RNG2 = np.random.default_rng(42)
    pos = int(RNG2.choice(valid))
    t_L, t_R = pos, pos + M_GAP + 1
    true = logp_t[t_L + 1:t_R]
    work = logp_t.copy(); work[t_L + 1:t_R] = np.nan
    mu_b, var_b = bivariate_bridge(work, logp_p, t_L, t_R, use_garch=True)
    sd_b = np.sqrt(var_b)
    k_idx = np.arange(t_L + 1, t_R)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    # LEFT: log-price view
    ax = axes[0]
    ax.plot(np.arange(t_L - 30, t_R + 30),
            logp_t[t_L - 30:t_R + 30], "k-", lw=0.7, label="observed log-price")
    ax.plot(k_idx, true, "C3-", lw=1.4, label="true (held out)")
    ax.fill_between(k_idx, mu_b - Z95 * sd_b, mu_b + Z95 * sd_b,
                    color="C1", alpha=0.3, label="bridge 95% CI")
    ax.plot(k_idx, mu_b, "C1-", lw=1.4, label="bridge estimate")
    ax.scatter([t_L, t_R], [logp_t[t_L], logp_t[t_R]],
               color="C0", zorder=5, s=25, label="endpoints (observed)")
    ax.set_xlabel("day"); ax.set_ylabel("log-price")
    ax.set_title(f"{a}: synthetic fold at day {pos}")
    ax.legend(fontsize=8, loc="best")
    # RIGHT: residual view (truth - estimate) vs. +/- 1.96 sd
    ax = axes[1]
    resid = true - mu_b
    ax.axhline(0, color="k", lw=0.6)
    ax.plot(k_idx, resid, "C3-", lw=1.4, label="truth $-$ estimate")
    ax.plot(k_idx, Z95 * sd_b, "C1--", lw=1.0, label="$\\pm 1.96\\,\\hat\\sigma_k$")
    ax.plot(k_idx, -Z95 * sd_b, "C1--", lw=1.0)
    ax.set_xlabel("day"); ax.set_ylabel("log-price residual")
    ax.set_title("Interior error relative to 95\\% band")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PIC / "fig_synth_fold_example.png", dpi=130)
    plt.close()
    print(f"Saved -> {PIC/'fig_synth_fold_example.png'}")


if __name__ == "__main__":
    main()
