"""Phase 4: signal evaluation of the processed factors (factors_z).

For each factor, at each monthly rebalance date t:
  rank IC(h) = Spearman( factor_t , return from close[t+1] to close[t+1+h] )
for h in HORIZONS trading days. Entry at t+1 so the signal is never traded on
the same close it was computed from.

Reported per factor:
  mean IC, std, IC t-stat = mean/std*sqrt(n) (guide's formula), Newey-West t
  (h > 21 makes monthly ICs overlap, so the naive t is overstated), hit rate
  (share of months IC > 0), decay curve over horizons, decile mean returns,
  top-minus-bottom spread and its monotonicity.

Also: the average cross-sectional Spearman correlation between factors.

Outputs: results/ic_summary.csv, ic_timeseries.csv, ic_decay.csv,
deciles.csv, factor_corr.csv, ic_by_year.csv; figures/ic_*.png, deciles.png,
factor_corr.png
"""
import numpy as np
import pandas as pd

from common import (FACTORS, FIGURES, PRIOR_SIGN, RESULTS, connect, diverging_cmap,
                    forward_returns, load_close, newey_west_t, style, SERIES, INK_2, MUTED)

HORIZONS = [5, 10, 21, 63]
MAIN_H = 21
N_DECILES = 10


def rank_ic(panel, factors, ret_col="fwd"):
    """Per-date Spearman IC of each factor against ret_col. Rows: dates."""
    def one(g):
        r = g[ret_col].rank()
        return pd.Series({f: g[f].rank().corr(r) for f in factors})
    return panel.groupby("rebal_date").apply(one, include_groups=False)


def summarize_ic(ic, horizon):
    lags = max(int(np.ceil(horizon / 21)) - 1, 0)
    rows = {}
    for f in ic.columns:
        x = ic[f].dropna()
        rows[f] = {"mean_ic": x.mean(), "std_ic": x.std(), "n": len(x),
                   "t_stat": x.mean() / x.std() * np.sqrt(len(x)),
                   "t_newey_west": newey_west_t(x, lags) if lags else x.mean() / x.std() * np.sqrt(len(x)),
                   "hit_rate": (x > 0).mean(), "icir_ann": x.mean() / x.std() * np.sqrt(12)}
    return pd.DataFrame(rows).T


def decile_returns(panel, factors, n=N_DECILES):
    """Mean forward return per factor decile (1 = lowest factor value), per date."""
    out = []
    for f in factors:
        d = panel[["rebal_date", f, "fwd"]].dropna()
        d["dec"] = d.groupby("rebal_date")[f].transform(
            lambda s: pd.qcut(s.rank(method="first"), n, labels=False) + 1)
        m = d.groupby(["rebal_date", "dec"])["fwd"].mean().unstack()
        m["factor"] = f
        out.append(m.reset_index())
    return pd.concat(out, ignore_index=True)


def monotonicity(mean_by_decile):
    """Spearman(decile, mean return) and share of adjacent steps in the expected direction."""
    v = np.asarray(mean_by_decile)
    rho = pd.Series(v).rank().corr(pd.Series(np.arange(len(v))).rank())
    steps = np.sign(np.diff(v))
    return rho, (steps == np.sign(rho)).mean()


def factor_corr(z, factors):
    mats = [g[factors].rank().corr() for _, g in z.groupby("rebal_date")]
    return sum(mats) / len(mats)


def main():
    con = connect()
    z = pd.read_sql("SELECT * FROM factors_z", con)
    raw = pd.read_sql("SELECT * FROM factors_raw", con)
    close = load_close(con)
    dates = sorted(z.rebal_date.unique())
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    # --- IC at each horizon
    decay, ic_main = {}, None
    for h in HORIZONS:
        fwd = forward_returns(close, dates, h)
        panel = z.merge(fwd, on=["rebal_date", "ticker"])
        ic = rank_ic(panel, FACTORS)
        s = summarize_ic(ic, h)
        decay[h] = s["mean_ic"]
        if h == MAIN_H:
            ic_main, summary, panel_main = ic, s, panel
    decay = pd.DataFrame(decay)
    decay.columns = [f"h{h}" for h in HORIZONS]

    # raw (not sector-neutral) IC for comparison
    fwd21 = forward_returns(close, dates, MAIN_H)
    ic_raw = rank_ic(raw.merge(fwd21, on=["rebal_date", "ticker"]), FACTORS)
    summary["mean_ic_raw"] = ic_raw.mean()
    summary["prior_sign"] = pd.Series(PRIOR_SIGN)
    summary["sign_agrees_with_prior"] = np.sign(summary.mean_ic) == summary.prior_sign

    # --- deciles
    dec = decile_returns(panel_main, FACTORS)
    dec_mean = dec.groupby("factor")[list(range(1, N_DECILES + 1))].mean().reindex(FACTORS)
    spread = dec.set_index(["factor", "rebal_date"])[N_DECILES] - dec.set_index(["factor", "rebal_date"])[1]
    summary["d10_minus_d1_monthly"] = spread.groupby("factor").mean()
    summary["d10_minus_d1_t"] = spread.groupby("factor").apply(lambda x: x.mean() / x.std() * np.sqrt(x.count()))
    mono = dec_mean.apply(lambda r: pd.Series(monotonicity(r.values), index=["mono_rho", "mono_steps"]), axis=1)
    summary = summary.join(mono)

    corr = factor_corr(z, FACTORS)
    by_year = ic_main.groupby(ic_main.index.str[:4]).mean()

    summary.round(4).to_csv(RESULTS / "ic_summary.csv", index_label="factor")
    ic_main.to_csv(RESULTS / "ic_timeseries.csv")
    decay.round(4).to_csv(RESULTS / "ic_decay.csv", index_label="factor")
    dec_mean.round(5).to_csv(RESULTS / "deciles.csv")
    corr.round(3).to_csv(RESULTS / "factor_corr.csv")
    by_year.round(4).to_csv(RESULTS / "ic_by_year.csv")

    pd.set_option("display.width", 200)
    print(f"Rebalance dates with 21d forward returns: {ic_main.shape[0]}")
    print(summary[["mean_ic", "t_stat", "hit_rate", "mean_ic_raw", "prior_sign",
                   "d10_minus_d1_monthly", "d10_minus_d1_t", "mono_rho"]].round(3).to_string())
    print("\nIC decay:\n", decay.round(4).to_string())
    print("\nDecile mean 21d returns (%):\n", (dec_mean * 100).round(2).to_string())
    print("\nFactor correlation:\n", corr.round(2).to_string())

    plot(summary, decay, dec_mean, corr, ic_main)


def plot(summary, decay, dec_mean, corr, ic):
    plt = style()

    # IC bars with t-stat labels
    fig, ax = plt.subplots(figsize=(7, 3.4))
    s = summary.sort_values("mean_ic")
    ax.barh(s.index, s.mean_ic, color=SERIES[0], height=0.55)
    ax.axvline(0, color=INK_2, lw=0.8)
    for i, (m, t) in enumerate(zip(s.mean_ic, s.t_stat)):
        ax.text(max(m, 0) + 0.001, i, f"t={t:.1f}", va="center", ha="left",
                fontsize=8, color=INK_2)
    ax.set_xlabel("Mean monthly rank IC (21-day forward return)")
    ax.set_title("Rank IC by factor, sector-neutral z-scores")
    ax.grid(axis="y", visible=False)
    fig.savefig(FIGURES / "ic_bars.png"); plt.close(fig)

    # decay small multiples
    fig, axes = plt.subplots(1, len(FACTORS), figsize=(12, 2.4), sharey=True)
    hs = [int(c[1:]) for c in decay.columns]
    for ax, f in zip(axes, FACTORS):
        ax.plot(range(len(hs)), decay.loc[f].values, color=SERIES[0], marker="o", ms=4)
        ax.axhline(0, color=INK_2, lw=0.8)
        ax.set_xticks(range(len(hs)), hs)
        ax.set_title(f, fontsize=9)
    axes[0].set_ylabel("Mean rank IC")
    fig.suptitle("IC decay: horizon in trading days", fontsize=11, color="#0b0b0b", y=1.04)
    fig.savefig(FIGURES / "ic_decay.png"); plt.close(fig)

    # deciles small multiples
    fig, axes = plt.subplots(1, len(FACTORS), figsize=(12, 2.6), sharey=True)
    for ax, f in zip(axes, FACTORS):
        v = dec_mean.loc[f].values * 100
        ax.bar(range(1, 11), v, color=SERIES[0], width=0.75)
        ax.axhline(0, color=INK_2, lw=0.8)
        ax.set_xticks([1, 5, 10])
        ax.set_title(f"{f}\nmono ρ={summary.loc[f, 'mono_rho']:.2f}", fontsize=9)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Mean 21d return, %")
    fig.suptitle("Mean forward return by factor decile (1 = lowest)", fontsize=11, color="#0b0b0b", y=1.08)
    fig.savefig(FIGURES / "deciles.png"); plt.close(fig)

    # correlation heatmap
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(corr.values, cmap=diverging_cmap(), vmin=-1, vmax=1)
    ax.set_xticks(range(len(FACTORS)), FACTORS, rotation=45, ha="right")
    ax.set_yticks(range(len(FACTORS)), FACTORS)
    ax.grid(False)
    for i in range(len(FACTORS)):
        for j in range(len(FACTORS)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="#0b0b0b" if abs(corr.values[i, j]) < 0.6 else "white")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Average cross-sectional rank correlation")
    fig.savefig(FIGURES / "factor_corr.png"); plt.close(fig)

    # cumulative IC
    fig, ax = plt.subplots(figsize=(8, 3.6))
    cum = ic.fillna(0).cumsum()
    idx = pd.to_datetime(cum.index)
    for i, f in enumerate(FACTORS):
        ax.plot(idx, cum[f], color=SERIES[i], lw=1.6, label=f)
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.legend(ncol=4, loc="upper left")
    ax.set_ylabel("Cumulative monthly IC")
    ax.set_title("Cumulative rank IC, 21-day horizon")
    fig.savefig(FIGURES / "ic_cumulative.png"); plt.close(fig)


if __name__ == "__main__":
    main()
