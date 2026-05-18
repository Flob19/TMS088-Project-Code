"""
Task 3 extension - Student-t final 200-day forecast table + PIT histograms
comparing Gaussian vs Student-t innovations.

Outputs:
  * LaTeX table (printed to stdout) - same style as tab:fc_final with extra
    nu column. Replaces 1.96 by scipy.stats.t.ppf(0.975, df=nu).
  * fig_pit_student_t.png   - RW-Gauss vs RW-t PIT histograms (7 panels).
  * fig_pit_arima_t.png     - ARIMA-Gauss vs ARIMA-t PIT histograms (7 panels).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from arch import arch_model

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from task2_interpolation import (
    load_clean, gap_bounds, ASSETS, USE_GARCH,
)
from task3_extrapolation import build_full_logp
from task3_validation import extrapolate_walkforward

HORIZON = 200
PIC_DIR = HERE / "Pictures"
PIC_DIR.mkdir(parents=True, exist_ok=True)
PIT_CSV = HERE / "task3_ext_pit_h200.csv"

ASSET_ORDER = ["gurkor", "water", "guitars", "slingshots",
               "stocks", "sugar", "tranquillity"]

VAR_LABEL = {a: ("GARCH(1,1)" if a in USE_GARCH else r"constant $\sigma$")
             for a in ASSET_ORDER}


def fit_nu_for_asset(logp_observed, asset):
    """For non-GARCH assets, fit nu on (r - mean)/std.  For GARCH assets,
    fit nu on standardised GARCH residuals r_t / sigma_t."""
    r = np.diff(logp_observed)
    r = r[~np.isnan(r)]
    if asset in USE_GARCH:
        try:
            am = arch_model(r * 100, vol="Garch", p=1, q=1,
                            mean="Zero", rescale=False, dist="normal")
            res = am.fit(disp="off", show_warning=False)
            sigma_t = np.asarray(res.conditional_volatility) / 100.0
            z = r / sigma_t
            z = z[np.isfinite(z)]
            z = (z - z.mean()) / z.std()
        except Exception:
            z = (r - r.mean()) / r.std()
    else:
        z = (r - r.mean()) / r.std()
    nu_fit = stats.t.fit(z, floc=0, fscale=1)[0]
    nu = float(np.clip(nu_fit, 3.0, 50.0))
    return nu


def build_table():
    df = load_clean()
    log_df = np.log(df[ASSETS])
    gaps = {a: gap_bounds(log_df[a]) for a in ASSETS}
    full_logp = build_full_logp(df, log_df, gaps)

    rows = []
    for a in ASSET_ORDER:
        logp = full_logp[a]
        last_obs_idx = int(np.where(~np.isnan(logp))[0][-1])
        logp_hist = logp[: last_obs_idx + 1]
        mean_log, cum_var = extrapolate_walkforward(logp_hist, a, HORIZON)
        mean_log_200 = float(mean_log[-1])
        s_200 = float(np.sqrt(cum_var[-1]))

        nu = fit_nu_for_asset(logp_hist, a)
        q = float(stats.t.ppf(0.975, df=nu))

        median_price = float(np.exp(mean_log_200))
        lo95 = float(np.exp(mean_log_200 - q * s_200))
        hi95 = float(np.exp(mean_log_200 + q * s_200))
        half_pct = (np.exp(q * s_200) - 1.0) * 100.0

        rows.append(dict(
            asset=a, variance=VAR_LABEL[a],
            median=median_price, lo=lo95, hi=hi95,
            half_pct=half_pct, nu=nu, q=q, s_200=s_200,
        ))
    return rows


def format_nu(nu):
    if nu >= 30.0:
        return "30+"
    return f"{nu:.1f}"


def print_latex(rows):
    header = ("% --- Student-t final 200-day forecast --- (drop into "
              "tab:fc_final with extra nu column)")
    lines = [header]
    name_w, var_w, med_w, band_w, half_w = 12, 18, 5, 14, 14
    for r in rows:
        band = f"[{r['lo']:.2f}, {r['hi']:.2f}]"
        half = f"$\\pm {r['half_pct']:.2f}\\%$"
        med = f"{r['median']:.2f}"
        lines.append(
            f"{r['asset']:<{name_w}} & {r['variance']:<{var_w}} & "
            f"{med:>{med_w}} & {band:<{band_w}} & {half:<{half_w}} & "
            f"{format_nu(r['nu'])} \\\\"
        )
    block = "\n".join(lines)
    print(block)
    return block


def plot_pit_panels(pit_df, model_g, model_t, suptitle, out_path):
    colors = plt.cm.tab10.colors
    cG, cT = colors[0], colors[3]

    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharex=True, sharey=True)
    axes = axes.ravel()
    for i, a in enumerate(ASSET_ORDER):
        ax = axes[i]
        uG = pit_df.loc[(pit_df.asset == a) & (pit_df.model == model_g), "u_pit"].to_numpy()
        uT = pit_df.loc[(pit_df.asset == a) & (pit_df.model == model_t), "u_pit"].to_numpy()
        ax.hist(uG, bins=10, range=(0, 1), density=True,
                color=cG, alpha=0.55, edgecolor="k",
                label=f"{model_g} (n={len(uG)})")
        ax.hist(uT, bins=10, range=(0, 1), density=True,
                color=cT, alpha=0.55, edgecolor="k",
                label=f"{model_t} (n={len(uT)})")
        ax.axhline(1.0, color="k", ls="--", lw=0.8)
        ax.set_title(a, fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, max(2.4, ax.get_ylim()[1]))
        if i % 4 == 0:
            ax.set_ylabel("density")
        if i >= 4:
            ax.set_xlabel("PIT value")
        ax.legend(fontsize=7, loc="upper center")
    for j in range(len(ASSET_ORDER), len(axes)):
        axes[j].axis("off")
    fig.suptitle(suptitle, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def make_figures():
    pit_df = pd.read_csv(PIT_CSV)
    out1 = PIC_DIR / "fig_pit_student_t.png"
    plot_pit_panels(
        pit_df, "RW-G", "RW-t",
        "PIT histograms: Gaussian vs Student-t innovations (h=200, RW+drift)",
        out1,
    )
    out2 = PIC_DIR / "fig_pit_arima_t.png"
    plot_pit_panels(
        pit_df, "ARIMA-G", "ARIMA-t",
        "PIT histograms: Gaussian vs Student-t innovations (h=200, ARIMA(1,1,1))",
        out2,
    )
    return out1, out2


def main():
    print("=" * 78)
    print("Student-t final 200-day forecast table")
    print("=" * 78)
    rows = build_table()
    print()
    print_latex(rows)

    print()
    print("Fitted nu per asset:")
    for r in rows:
        print(f"  {r['asset']:14s}  nu = {r['nu']:.3f}   "
              f"q_975 = {r['q']:.3f}   S_200 = {r['s_200']:.4f}")

    print()
    print("Generating PIT figures...")
    out1, out2 = make_figures()
    sz1 = out1.stat().st_size
    sz2 = out2.stat().st_size
    print(f"  saved {out1}  ({sz1:,} bytes)")
    print(f"  saved {out2}  ({sz2:,} bytes)")


if __name__ == "__main__":
    main()
