r"""
Task 4 — Investment strategies.

Four strategies, evaluated on the cleaned + interpolated price history (the
trailing 200 days are not used; we keep them unseen for Task 3 only).

Strategies:
  1. Buy & Hold (equal-weight, full range)   -- passive baseline
  2. Moving-average crossover (per asset, equal-weight long positions)
  3. Inverse-volatility risk parity (full range, monthly rebalance)
  4. Cross-sectional momentum (full range, hold top 3, monthly rebalance)

Performance measured on daily log-returns, annualised by sqrt(252).  Sharpe
uses a risk-free rate of 3% per year.  We report:
  * annualised return
  * annualised volatility
  * Sharpe ratio
  * maximum drawdown
  * terminal equity (starting from 1)

We also split the sample in two halves (design / OOS) and report Sharpe in
each half to guard against over-fitting.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from task2_interpolation import (
    load_clean, gap_bounds, univariate_bridge, bivariate_bridge,
    ASSETS, PEERS, USE_GARCH, PIC,
)

ROOT = Path(__file__).parent
RF_ANNUAL = 0.03
DAYS_PER_YEAR = 252
RF_DAILY = RF_ANNUAL / DAYS_PER_YEAR


# ---------------------------------------------------------------------------
# Build full clean price panel with the interpolated gap filled in
# ---------------------------------------------------------------------------
def build_price_panel():
    df = load_clean()
    log_df = np.log(df[ASSETS])
    gaps = {a: gap_bounds(log_df[a]) for a in ASSETS}
    filled = {}
    for a in ASSETS:
        logp = log_df[a].to_numpy().copy()
        t_L, t_R = gaps[a]
        peer = PEERS.get(a)
        use_garch = a in USE_GARCH
        if peer is not None:
            mu, _ = bivariate_bridge(logp, log_df[peer].to_numpy(),
                                     t_L, t_R, use_garch=use_garch)
        else:
            mu, _ = univariate_bridge(logp, t_L, t_R, use_garch=use_garch)
        logp[t_L + 1:t_R] = mu
        # fill remaining isolated outlier NaNs (5 single-day spikes from
        # Task 1 cleaning) by linear interpolation on log-prices -- they are
        # not contiguous gaps so a straight line is harmless here
        s = pd.Series(logp).interpolate(method="linear",
                                        limit_direction="both").to_numpy()
        filled[a] = s
    # drop trailing NaNs
    mat = np.column_stack([filled[a] for a in ASSETS])
    last_idx = np.where(~np.isnan(mat).any(axis=1))[0][-1]
    mat = mat[:last_idx + 1]
    prices = pd.DataFrame(np.exp(mat), columns=ASSETS)
    return prices


# ---------------------------------------------------------------------------
# Strategy implementations.  Each returns a pd.Series of daily portfolio
# log-returns indexed by day.
# ---------------------------------------------------------------------------
def strat_buy_and_hold(prices):
    """Equal weight at t=0, let positions drift (no rebalancing)."""
    p = prices.values
    # shares bought at t=0 with 1/7 capital each at initial price
    init = p[0]
    shares = (1.0 / len(ASSETS)) / init
    portfolio = p @ shares
    log_r = np.diff(np.log(portfolio))
    return pd.Series(log_r, index=prices.index[1:])


def strat_ma_crossover(prices, short=20, long=100):
    """Per-asset 20/100 MA crossover: hold 1/N long if short>long else 0.
    Signal uses data up to t-1 (no look-ahead)."""
    r = np.log(prices).diff()
    short_ma = prices.rolling(short).mean()
    long_ma = prices.rolling(long).mean()
    signal = (short_ma > long_ma).astype(float).shift(1)  # trade on prior close
    # equal-weight across active signals; cash gets risk-free return
    w = signal.div(len(ASSETS))
    port_r = (w * r).sum(axis=1)
    # cash weight = 1 - sum(w); earns RF_DAILY
    cash_w = 1.0 - w.sum(axis=1)
    port_r = port_r + cash_w * RF_DAILY
    return port_r.dropna()


def strat_inverse_vol(prices, vol_window=60, rebal_every=21):
    """Risk parity: w_i ~ 1/sigma_i, rebalance monthly. Full range, always
    100% invested."""
    r = np.log(prices).diff()
    vol = r.rolling(vol_window).std()
    inv_vol = 1.0 / vol
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)
    # freeze weights between rebalance dates (monthly)
    idx = np.arange(len(weights))
    rebal_mask = (idx % rebal_every == 0)
    frozen = weights.copy()
    last = None
    out = np.full_like(weights.values, np.nan)
    for i in range(len(weights)):
        if rebal_mask[i] and not weights.iloc[i].isna().any():
            last = weights.iloc[i].values
        if last is not None:
            out[i] = last
    w = pd.DataFrame(out, index=weights.index, columns=ASSETS).shift(1)
    port_r = (w * r).sum(axis=1)
    return port_r.dropna()


def strat_channel_breakout(prices, lookback=55, exit_lookback=20):
    """Donchian-style channel breakout per asset.  Long a 1/7 position when
    today's close exceeds the max of the prior `lookback` closes (signal
    observed yesterday); exit to cash when today's close falls below the
    min of the prior `exit_lookback` closes.  Cash earns RF_DAILY."""
    r = np.log(prices).diff()
    p = prices.values
    n, m = p.shape
    pos = np.zeros((n, m))
    for i in range(max(lookback, exit_lookback) + 1, n):
        hi = p[i - lookback:i].max(axis=0)
        lo = p[i - exit_lookback:i].min(axis=0)
        for j in range(m):
            if pos[i - 1, j] == 0 and p[i - 1, j] > hi[j] * (1 - 1e-12):
                pos[i, j] = 1.0
            elif pos[i - 1, j] == 1 and p[i - 1, j] < lo[j] * (1 + 1e-12):
                pos[i, j] = 0.0
            else:
                pos[i, j] = pos[i - 1, j]
    w = pd.DataFrame(pos / len(ASSETS), index=prices.index, columns=ASSETS)
    port_r = (w * r).sum(axis=1)
    cash_w = 1.0 - w.sum(axis=1)
    port_r = port_r + cash_w * RF_DAILY
    return port_r.dropna()


def strat_leadlag_sugar(prices):
    """Task 1-motivated lead-lag strategy.  Task 1 found rho(guitars_{t-1},
    sugar_t) ~= 0.36 and rho(slingshots_{t-1}, sugar_t) ~= 0.39 -- i.e.
    guitars and slingshots LEAD sugar by one day.  Go long sugar (weight
    1/7) only when the previous day's guitars+slingshots average return
    was positive; otherwise hold cash.  The remaining six assets are held
    equal-weight (1/7 each)."""
    r = np.log(prices).diff()
    signal = 0.5 * (r["guitars"].shift(1) + r["slingshots"].shift(1))
    w = pd.DataFrame(0.0, index=prices.index, columns=ASSETS)
    for a in ASSETS:
        if a != "sugar":
            w[a] = 1.0 / 7
    w["sugar"] = (signal > 0).astype(float) * (1.0 / 7)
    port_r = (w * r).sum(axis=1)
    cash_w = 1.0 - w.sum(axis=1)
    port_r = port_r + cash_w * RF_DAILY
    return port_r.dropna()


def strat_garch_timing(prices, rebal_every=5, fit_window=1000, warmup=500):
    """GARCH-based volatility timing: inverse-vol weights using a walk-forward
    GARCH(1,1) conditional-volatility estimate per asset (rather than rolling
    60-day std as in strat_inverse_vol).  At each rebalance day, re-fit
    GARCH(1,1) per asset on the trailing `fit_window` days of log-returns and
    use the one-step-ahead conditional-volatility forecast as the weight
    input.  This removes the parameter look-ahead of fitting once on the full
    sample.  Full range, always 100% invested.  `warmup` is the minimum
    history required before the first rebalance."""
    from arch import arch_model
    r = np.log(prices).diff()
    n, m_ = prices.shape
    out = np.full((n, m_), np.nan)
    last = None
    for i in range(n):
        if i >= warmup and (i % rebal_every == 0):
            sigmas = np.empty(m_)
            for j, a in enumerate(ASSETS):
                lo = max(0, i - fit_window)
                y = r[a].iloc[lo:i].dropna().values * 100
                if len(y) < 100:
                    sigmas[j] = np.nanstd(y) / 100
                    continue
                try:
                    am = arch_model(y, vol="Garch", p=1, q=1, mean="Zero",
                                    rescale=False, dist="normal")
                    res = am.fit(disp="off", show_warning=False)
                    v_next = res.forecast(horizon=1, reindex=False)\
                                .variance.values[-1][0]
                    sigmas[j] = np.sqrt(v_next) / 100
                except Exception:
                    sigmas[j] = np.nanstd(y) / 100
            inv_vol = 1.0 / np.clip(sigmas, 1e-6, None)
            last = inv_vol / inv_vol.sum()
        if last is not None:
            out[i] = last
    w = pd.DataFrame(out, index=prices.index, columns=ASSETS).shift(1)
    port_r = (w * r).sum(axis=1)
    return port_r.dropna()


def strat_short_stocks(prices, hedge_weight=0.5):
    """Task 1-motivated asymmetric strategy: go long the full range at
    equal weight, and short `hedge_weight` of stocks against it.  Designed
    to test whether the negative historical drift of stocks is
    exploitable."""
    r = np.log(prices).diff()
    n_assets = len(ASSETS)
    w = pd.DataFrame(1.0 / n_assets, index=prices.index, columns=ASSETS)
    w["stocks"] = -hedge_weight
    port_r = (w * r).sum(axis=1)
    return port_r.dropna()


def strat_buy_dips(prices, vol_window=20, vol_thresh_mult=1.5,
                   ref_window=250, recover_window=20):
    """Volatility-conditioned buy-the-dip strategy.

    Asset categorisation follows Task 1 volatilities:
      * LOW_VOL   = {water, gurkor}                — 1/7 long, always
      * HIGH_VOL  = {tranquillity, slingshots, guitars, stocks, sugar}
                                                   — 1/14 long, always
                                                   — +1/14 dip overlay when triggered

    Dip-entry signal (per HIGH_VOL asset):
      * yesterday's `vol_window`-day rolling std > `vol_thresh_mult`
        times its rolling `ref_window`-day mean (= "vol is elevated"), AND
      * yesterday's log-return < 0 (= "yesterday was a down day").

    Dip-exit signal:
      * today's price reaches its `recover_window`-day rolling high
        (= "asset has recovered"), at which point the +1/14 overlay
        is removed and the asset returns to the 1/14 baseline.

    Default exposure (no dips active) = 2/7 + 5/14 = 9/14 ≈ 64% long.
    Max exposure (all 5 dips active) = 2/7 + 10/14 = 1.0.
    Remainder is cash earning RF_DAILY.

    All signals use lagged data (yesterday) so there is no look-ahead.
    """
    LOW_VOL = ["water", "gurkor"]
    HIGH_VOL = ["tranquillity", "slingshots", "guitars", "stocks", "sugar"]

    r = np.log(prices).diff()
    vol_short = r.rolling(vol_window).std()
    vol_ref = vol_short.rolling(ref_window).mean()
    vol_thresh = vol_ref * vol_thresh_mult
    high_recover = prices.rolling(recover_window).max()

    n = len(prices)
    asset_idx = {a: i for i, a in enumerate(ASSETS)}
    weights = np.zeros((n, len(ASSETS)))

    in_dip = {a: False for a in HIGH_VOL}

    for i in range(n):
        # baseline weights for day i
        row = np.zeros(len(ASSETS))
        for a in LOW_VOL:
            row[asset_idx[a]] = 1.0 / 7
        for a in HIGH_VOL:
            row[asset_idx[a]] = 1.0 / 14

        if i >= 1:
            # update dip state per high-vol asset based on yesterday's data
            for a in HIGH_VOL:
                v_prev = vol_short[a].iloc[i - 1]
                v_thresh_prev = vol_thresh[a].iloc[i - 1]
                r_prev = r[a].iloc[i - 1]
                p_prev = prices[a].iloc[i - 1]
                high_prev = high_recover[a].iloc[i - 1]

                if (pd.isna(v_prev) or pd.isna(v_thresh_prev)
                        or pd.isna(high_prev) or pd.isna(r_prev)):
                    continue

                if not in_dip[a]:
                    if v_prev > v_thresh_prev and r_prev < 0:
                        in_dip[a] = True
                else:
                    # exit when price recovers to 20-day high
                    if p_prev >= high_prev * (1 - 1e-9):
                        in_dip[a] = False

                if in_dip[a]:
                    row[asset_idx[a]] = 1.0 / 14 + 1.0 / 14

        weights[i] = row

    w = pd.DataFrame(weights, index=prices.index, columns=ASSETS)
    port_r = (w * r).sum(axis=1)
    cash_w = 1.0 - w.sum(axis=1)
    port_r = port_r + cash_w * RF_DAILY
    return port_r.dropna()


def strat_momentum(prices, lookback=60, top_k=3, rebal_every=21):
    """Cross-sectional momentum: hold top-K by trailing 60d return, equal
    weight, monthly rebalance, flat in cash otherwise."""
    r = np.log(prices).diff()
    mom = np.log(prices) - np.log(prices.shift(lookback))
    # on each rebal day: rank and take top K
    w = pd.DataFrame(0.0, index=prices.index, columns=ASSETS)
    idx = np.arange(len(prices))
    rebal_mask = (idx % rebal_every == 0)
    last_w = np.zeros(len(ASSETS))
    for i in range(len(prices)):
        if rebal_mask[i] and not mom.iloc[i].isna().any():
            ranks = mom.iloc[i].rank(ascending=False)
            new = np.zeros(len(ASSETS))
            new[ranks.values <= top_k] = 1.0 / top_k
            last_w = new
        w.iloc[i] = last_w
    w = w.shift(1)
    port_r = (w * r).sum(axis=1)
    cash_w = 1.0 - w.sum(axis=1)
    port_r = port_r + cash_w * RF_DAILY
    return port_r.dropna()


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------
def metrics(log_ret):
    """log_ret: pd.Series of daily log-returns."""
    r = log_ret.values
    n = len(r)
    ann_ret = r.mean() * DAYS_PER_YEAR
    ann_vol = r.std(ddof=1) * np.sqrt(DAYS_PER_YEAR)
    sharpe = (ann_ret - RF_ANNUAL) / ann_vol if ann_vol > 0 else np.nan
    # Lo (2002) approximation: SE(Sharpe) ~= sqrt((1 + S^2/2) / N_years)
    n_years = n / DAYS_PER_YEAR
    se = np.sqrt((1 + 0.5 * (sharpe ** 2)) / n_years) if np.isfinite(sharpe) else np.nan
    sharpe_lo = sharpe - 1.96 * se
    sharpe_hi = sharpe + 1.96 * se
    equity = np.exp(np.cumsum(r))
    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1
    max_dd = drawdown.min()
    return dict(
        n=n,
        ann_return=ann_ret,
        ann_vol=ann_vol,
        sharpe=sharpe,
        sharpe_lo=sharpe_lo,
        sharpe_hi=sharpe_hi,
        max_drawdown=max_dd,
        terminal_equity=float(equity[-1]),
    )


def split_metrics(log_ret, split=0.5):
    cut = int(len(log_ret) * split)
    return metrics(log_ret.iloc[:cut]), metrics(log_ret.iloc[cut:])


def main():
    prices = build_price_panel()
    print(f"Price panel: {len(prices)} days  x {len(ASSETS)} assets")

    strategies = {
        "Buy & Hold (EW)":          strat_buy_and_hold(prices),
        "MA crossover 20/100":      strat_ma_crossover(prices),
        "Inverse-vol risk parity":  strat_inverse_vol(prices),
        "Momentum top-3":           strat_momentum(prices),
        "Channel breakout 55/20":   strat_channel_breakout(prices),
        "Buy dips (vol-conditioned)": strat_buy_dips(prices),
        "Short stocks (hedged)":    strat_short_stocks(prices),
        "Lead-lag sugar (Task 1)":  strat_leadlag_sugar(prices),
        "GARCH vol timing":         strat_garch_timing(prices),
    }

    # Performance table (full sample + OOS second half)
    rows = []
    for name, r in strategies.items():
        full = metrics(r)
        design, oos = split_metrics(r, split=0.5)
        rows.append({
            "strategy": name,
            "n_days": full["n"],
            "ann_return_%": 100 * full["ann_return"],
            "ann_vol_%": 100 * full["ann_vol"],
            "sharpe": full["sharpe"],
            "sharpe_95ci": f"[{full['sharpe_lo']:+.2f}, {full['sharpe_hi']:+.2f}]",
            "max_drawdown_%": 100 * full["max_drawdown"],
            "terminal_equity": full["terminal_equity"],
            "sharpe_firsthalf": design["sharpe"],
            "sharpe_secondhalf": oos["sharpe"],
        })
    perf = pd.DataFrame(rows)
    print("\nStrategy performance:")
    print(perf.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    perf.to_csv(ROOT / "task4_strategies.csv", index=False)
    print(f"Saved -> {ROOT/'task4_strategies.csv'}")

    # ---------- equity curves (single panel, all strategies) ----------
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, r in strategies.items():
        eq = np.exp(np.cumsum(r.values))
        ax.plot(r.index, eq, label=name, lw=1.2)
    ax.set_yscale("log")
    ax.set_title("Equity curves (log scale, start = 1)")
    ax.set_xlabel("day"); ax.set_ylabel("portfolio value")
    ax.legend(fontsize=9, loc="upper left"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PIC / "fig_strategy_equity.png", dpi=130)
    plt.close()
    print(f"Saved -> {PIC/'fig_strategy_equity.png'}")

    # ---------- drawdowns as small multiples (one panel per strategy) ----------
    n_strat = len(strategies)
    ncols = 3
    nrows = int(np.ceil(n_strat / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.0 * nrows),
                             sharex=True, sharey=True)
    axes = axes.ravel()
    colors = plt.cm.tab10.colors
    for i, (name, r) in enumerate(strategies.items()):
        eq = np.exp(np.cumsum(r.values))
        dd = (eq / np.maximum.accumulate(eq) - 1) * 100
        ax = axes[i]
        ax.fill_between(r.index, dd, 0, color=colors[i], alpha=0.45)
        ax.plot(r.index, dd, color=colors[i], lw=0.8)
        ax.axhline(0, color="k", lw=0.5)
        max_dd = dd.min()
        ax.set_title(f"{name}  (max DD = {max_dd:+.1f}\\%)", fontsize=10)
        ax.set_ylabel("drawdown (%)"); ax.grid(alpha=0.3)
        ax.set_xlabel("day")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig(PIC / "fig_strategy_drawdowns.png", dpi=130)
    plt.close()
    print(f"Saved -> {PIC/'fig_strategy_drawdowns.png'}")

    # ---------- split-sample sharpes ----------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = [r["strategy"] for r in rows]
    firsthalf = [r["sharpe_firsthalf"] for r in rows]
    secondhalf = [r["sharpe_secondhalf"] for r in rows]
    x = np.arange(len(names))
    ax.bar(x - 0.2, firsthalf, 0.4, label="first half", color="C0")
    ax.bar(x + 0.2, secondhalf, 0.4, label="second half", color="C1")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("annualised Sharpe (rf = 3%)")
    ax.set_title("Split-sample Sharpe: first vs. second half of the price history")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(PIC / "fig_strategy_sharpe_split.png", dpi=130)
    plt.close()
    print(f"Saved -> {PIC/'fig_strategy_sharpe_split.png'}")

    # ---------- momentum robustness grid ----------
    print("\nMomentum robustness grid (Sharpe ratio):")
    lookbacks = [30, 60, 90, 120]
    ks = [2, 3, 4]
    grid_rows = []
    for lb in lookbacks:
        for k in ks:
            r = strat_momentum(prices, lookback=lb, top_k=k)
            m_ = metrics(r)
            grid_rows.append({"lookback": lb, "top_k": k,
                              "sharpe": m_["sharpe"],
                              "max_dd_%": 100 * m_["max_drawdown"]})
    grid = pd.DataFrame(grid_rows)
    pivot = grid.pivot(index="lookback", columns="top_k", values="sharpe")
    print(pivot.round(3).to_string())
    grid.to_csv(ROOT / "task4_momentum_grid.csv", index=False)
    print(f"Saved -> {ROOT/'task4_momentum_grid.csv'}")


if __name__ == "__main__":
    main()
