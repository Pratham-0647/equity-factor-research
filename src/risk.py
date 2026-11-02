"""Phase 7: risk metrics, residual beta, robustness charts, variant ledger.

Reads results/bt_daily.csv (daily net returns, base 20 bps costs).

Metrics per variant: CAGR, annualised vol, Sharpe, Sortino, max drawdown,
Calmar, monthly hit rate, worst month, skew; residual market beta = OLS slope
of daily returns on Nifty 50 daily returns with HAC (Newey-West, 5 lags) t-stat.
Long-only Sharpe/Sortino use excess return over RF_ANNUAL; long-short books are
self-financing and use raw returns.

  Sharpe  = mean(excess) / std(r)                  * sqrt(252)
  Sortino = mean(excess) / sqrt(mean(min(excess,0)^2)) * sqrt(252)
  Calmar  = CAGR / |max drawdown|

Multiple testing: every variant evaluated on out-of-sample performance is
listed in results/variants.csv. With N trials, the best Sharpe of N
no-skill strategies has expectation ~ sd * E[max of N std normals], sd ~ 1/sqrt(years).
The deflated Sharpe probability is P(true Sharpe > that hurdle). This is the
simple annualised form of Bailey & Lopez de Prado (2014); N counts correlated
variants as independent, so the hurdle is conservative.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

from backtest import RF_ANNUAL
from common import FIGURES, INK_2, RESULTS, SERIES, diverging_cmap, style

HEADLINE = "lgbm|ls|ew|monthly"
LS_SHOW = [("lgbm|ls|ew|monthly", "LightGBM L/S"), ("linear|ls|ew|monthly", "Linear L/S"),
           ("ew_prior|ls|ew|monthly", "EW composite L/S")]


def max_drawdown(r):
    nav = (1 + r).cumprod()
    dd = nav / nav.cummax() - 1
    return dd.min(), dd


def metrics(r, mkt, rf=0.0):
    r = r.dropna()
    yrs = len(r) / 252
    ex = r - rf / 252
    cagr = (1 + r).prod() ** (1 / yrs) - 1
    mdd, _ = max_drawdown(r)
    monthly = (1 + r).groupby(pd.to_datetime(r.index).to_period("M")).prod() - 1
    X = sm.add_constant(mkt.reindex(r.index).fillna(0).values)
    fit = sm.OLS(r.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return {"cagr": cagr, "vol": r.std() * np.sqrt(252),
            "sharpe": ex.mean() / r.std() * np.sqrt(252),
            "sortino": ex.mean() / np.sqrt((np.minimum(ex, 0) ** 2).mean()) * np.sqrt(252),
            "max_drawdown": mdd, "calmar": cagr / abs(mdd),
            "hit_rate_monthly": (monthly > 0).mean(), "worst_month": monthly.min(),
            "best_month": monthly.max(), "skew_daily": r.skew(),
            "market_beta": fit.params[1], "beta_t": fit.tvalues[1],
            "alpha_ann": fit.params[0] * 252, "years": yrs}


def expected_max_sr(n, sd):
    g = 0.5772156649
    if n <= 1:
        return 0.0
    return sd * ((1 - g) * norm.ppf(1 - 1 / n) + g * norm.ppf(1 - 1 / (n * np.e)))


def variant_ledger():
    """Every configuration whose out-of-sample performance was looked at."""
    rows = [
        ("Phase 4", "factor x horizon IC evaluations", 7 * 4, "7 factors x {5,10,21,63}d"),
        ("Phase 4", "raw vs sector-neutral IC", 7, "diagnostic"),
        ("Phase 5", "LightGBM hyperparameter configs (inner CV only)", 8, "chosen on training folds, never on OOS"),
        ("Phase 5", "OOS signal models", 5, "lgbm, linear, ew_prior, ew_ic, best_1"),
        ("Phase 6", "backtest variants", 15, "3 signals x L/S,L/O x EW,vol + 2 weekly + universe"),
        ("Phase 6", "cost levels in sweep", 21, "sensitivity, not selection"),
        ("Phase 7", "lookback grid cells (full refit)", 16, "momentum 4 x vol window 4"),
        ("Phase 7", "portfolio grid cells", 16, "quantile 4 x cost 4"),
    ]
    led = pd.DataFrame(rows, columns=["phase", "what", "count", "note"])
    # strategy variants whose Sharpe could have been cherry-picked
    n_strat = 5 + 15 + 16 + 16
    return led, n_strat


def main():
    daily = pd.read_csv(RESULTS / "bt_daily.csv", index_col="date")
    mkt = daily.pop("nifty50")
    rows = {}
    for col in daily.columns:
        rf = RF_ANNUAL if "|lo|" in col else 0.0
        rows[col] = metrics(daily[col], mkt, rf)
    rows["nifty50"] = metrics(mkt, mkt, RF_ANNUAL)
    rows["nifty50"].update(market_beta=1.0, beta_t=np.nan, alpha_ann=0.0)
    rows["lgbm_lo_minus_universe"] = metrics(daily["lgbm|lo|ew|monthly"] - daily["universe|lo|ew|monthly"], mkt)
    m = pd.DataFrame(rows).T
    bt = pd.read_csv(RESULTS / "bt_summary.csv", index_col="variant")
    m = m.join(bt[["turnover_per_rebal", "turnover_annual", "cost_drag_annual"]])

    led, n = variant_ledger()
    sd = 1 / np.sqrt(m.loc[HEADLINE, "years"])
    hurdle = expected_max_sr(n, sd)
    sr = m.loc[HEADLINE, "sharpe"]
    dsr = pd.DataFrame([{"variant": HEADLINE, "sharpe_net": sr, "n_strategy_variants": n,
                         "sharpe_sd": sd, "expected_max_null_sharpe": hurdle,
                         "deflated_sharpe_prob": norm.cdf((sr - hurdle) / sd),
                         "plain_sharpe_prob": norm.cdf(sr / sd)}])
    for v in ["linear|ls|ew|monthly"]:
        s2 = m.loc[v, "sharpe"]
        dsr.loc[len(dsr)] = [v, s2, n, sd, hurdle, norm.cdf((s2 - hurdle) / sd), norm.cdf(s2 / sd)]

    m.round(4).to_csv(RESULTS / "risk_metrics.csv", index_label="variant")
    led.to_csv(RESULTS / "variants.csv", index=False)
    dsr.round(4).to_csv(RESULTS / "deflated_sharpe.csv", index=False)

    pd.set_option("display.width", 220)
    cols = ["cagr", "vol", "sharpe", "sortino", "max_drawdown", "calmar", "hit_rate_monthly",
            "market_beta", "beta_t", "turnover_annual"]
    print(m[cols].round(3).to_string())
    print(f"\nVariants: {led['count'].sum()} configurations logged; {n} strategy variants scored on OOS performance")
    print(dsr.round(3).to_string(index=False))
    plot(daily, mkt)


def plot(daily, mkt):
    plt = style()
    idx = pd.to_datetime(daily.index)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for i, (c, lab) in enumerate(LS_SHOW):
        r = daily[c]
        roll = r.rolling(252).mean() / r.rolling(252).std() * np.sqrt(252)
        axes[0].plot(idx, roll, color=SERIES[i], lw=1.5, label=lab)
        _, dd = max_drawdown(r)
        axes[1].plot(idx, dd * 100, color=SERIES[i], lw=1.5, label=lab)
    axes[0].axhline(0, color=INK_2, lw=0.8)
    axes[0].set_ylabel("Rolling 12-month Sharpe")
    axes[0].set_title("Rolling Sharpe and drawdown, long-short net of 20 bps")
    axes[0].legend(loc="upper left", ncol=3)
    axes[1].set_ylabel("Drawdown, %")
    fig.savefig(FIGURES / "rolling_sharpe_drawdown.png"); plt.close(fig)

    r = daily[HEADLINE]
    mo = (1 + r).groupby(pd.to_datetime(r.index).to_period("M")).prod() - 1
    tab = pd.DataFrame({"y": mo.index.year, "m": mo.index.month, "v": mo.values * 100}).pivot(
        index="y", columns="m", values="v")
    fig, ax = plt.subplots(figsize=(9, 4))
    v = np.nanmax(np.abs(tab.values))
    im = ax.imshow(tab.values, cmap=diverging_cmap().reversed(), vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(12), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_yticks(range(len(tab)), tab.index)
    ax.grid(False)
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            x = tab.values[i, j]
            if not np.isnan(x):
                ax.text(j, i, f"{x:.1f}", ha="center", va="center", fontsize=7, color="#0b0b0b")
    ax.set_title("LightGBM long-short monthly net return, % (blue = gain, red = loss)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(FIGURES / "monthly_heatmap.png"); plt.close(fig)


if __name__ == "__main__":
    main()
