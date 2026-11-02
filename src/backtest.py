"""Phase 6: portfolio construction and daily backtest, net of costs.

Timing: a signal formed at close t is traded at close t+1 (ENTRY_LAG). Holdings
then drift with prices until the next rebalance. Each day the existing book
earns that day's return first; on a trade day it is then rebalanced at the close.

Book accounting: capital (NAV) starts at 1. Long-short books are dollar-neutral,
+1 x NAV long and -1 x NAV short (gross 2), so a period's return is close to the
top-minus-bottom decile spread. Long-only books are +1 x NAV.

Costs: each rebalance pays (cost_bps + slippage_bps) / 1e4 on traded notional,
i.e. on one-way turnover = sum |target - current| / NAV. Base case 15 + 5 bps
(Indian delivery STT 10 bps per side plus stamp duty, exchange and brokerage
fees, plus slippage for large/mid caps). The short side is assumed implementable
via single-stock futures; borrow/financing costs are not modelled.

Signals (out-of-sample from Phase 5, 2019 onward): lgbm (headline, chosen before
any backtest was run), linear, ew_prior. Weekly rebalancing is tested on the
ew_prior composite only, because the ML models are trained on monthly samples.

Outputs: results/bt_daily.csv (daily net/gross returns of every variant),
results/bt_summary.csv, results/bt_cost_sweep.csv, figures/cost_sensitivity.png,
figures/equity_curves.png
"""
import numpy as np
import pandas as pd

import factors as F
import preprocess as P
from common import (ENTRY_LAG, FACTORS, FFILL_LIMIT, FIGURES, PRIOR_SIGN, RESULTS, SERIES,
                    INK_2, connect, load_close, load_market, style)

BASE_COST_BPS = 15.0
BASE_SLIP_BPS = 5.0
QUANTILE = 0.10
RF_ANNUAL = 0.06          # approx. Indian 91-day T-bill, 2019-2026
COST_GRID = list(range(0, 105, 5))


# ------------------------------------------------------------------ engine
def daily_returns(close):
    lvl = close.ffill(limit=FFILL_LIMIT)
    return lvl.pct_change(fill_method=None).fillna(0.0)


def target_weights(sig, col, side, scheme="ew", q=QUANTILE, vol=None):
    """Weights for one date. side: 'ls' (long top q, short bottom q) or 'lo'."""
    s = sig[["ticker", col]].dropna()
    n = max(int(round(len(s) * q)), 1)
    r = s[col].rank(method="first")
    top = s.loc[r > len(s) - n, "ticker"]
    bot = s.loc[r <= n, "ticker"]

    def leg(names):
        if scheme == "vol":
            iv = 1.0 / vol.reindex(names)
            iv = iv.fillna(iv.median())
            return iv / iv.sum()
        return pd.Series(1.0 / len(names), index=names.values)

    w = leg(top)
    if side == "ls":
        w = w.sub(leg(bot), fill_value=0.0)
    return w


def run(weights, rets, cost_bps):
    """Simulate. weights: {signal_date: Series}. Returns daily DataFrame."""
    days = rets.index
    exec_at = {}
    for d, w in weights.items():
        i = days.get_loc(d) + ENTRY_LAG
        if i < len(days):
            exec_at[days[i]] = w
    first = min(exec_at)
    tickers = rets.columns
    R = rets.loc[first:].to_numpy()
    idx = rets.loc[first:].index
    h = np.zeros(len(tickers))
    nav = 1.0
    out = []
    for k, d in enumerate(idx):
        r = R[k]
        pre = nav + h @ r
        h = h * (1 + r)
        turn = cost = 0.0
        if d in exec_at:
            tgt = exec_at[d].reindex(tickers).fillna(0.0).to_numpy() * pre
            turn = np.abs(tgt - h).sum() / pre
            cost = turn * cost_bps / 1e4 * pre
            h = tgt * (pre - cost) / pre
        new = pre - cost
        out.append((d, new / nav - 1, turn, cost / nav))
        nav = new
    return pd.DataFrame(out, columns=["date", "ret", "turnover", "cost"]).set_index("date")


# ------------------------------------------------------------------ signals
def weekly_signals(con, close, start):
    """ew_prior composite on the last trading day of each week (from `start`).

    Universe = latest month-end universe on or before the week date (causal).
    Liquidity = that month-end's median traded value (up to 4 weeks stale).
    """
    days = pd.Series(close.index, index=pd.to_datetime(close.index))
    wk = days.groupby(days.index.to_period("W")).max()
    wk = [d for d in wk.values if d >= start]
    uni = pd.read_sql("SELECT rebal_date, ticker, median_tv FROM universe", con)
    month_ends = np.array(sorted(uni.rebal_date.unique()))
    mp = {d: month_ends[month_ends <= d][-1] for d in wk}
    u = pd.DataFrame({"rebal_date": list(mp), "month_end": list(mp.values())}).merge(
        uni.rename(columns={"rebal_date": "month_end"}), on="month_end").drop(columns="month_end")
    panels = F.compute_factor_panels(close, load_market(con, close.index))
    sectors = pd.read_sql("SELECT ticker, sector FROM sectors", con)
    raw = F.sample_at_rebalance(panels, u, sectors)
    z = P.process(raw)
    z["ew_prior"] = sum(z[f].fillna(0) * s for f, s in PRIOR_SIGN.items()) / len(PRIOR_SIGN)
    return z


def build_weights(sig, col, side, scheme, vol_lookup):
    out = {}
    for d, g in sig.groupby("rebal_date"):
        vol = vol_lookup.get(d) if scheme == "vol" else None
        out[d] = target_weights(g, col, side, scheme, vol=vol)
    return out


def main():
    con = connect()
    close = load_close(con)
    rets = daily_returns(close)
    mkt = load_market(con, close.index).ffill().pct_change(fill_method=None).fillna(0.0)
    preds = pd.read_sql("SELECT * FROM predictions", con)
    raw = pd.read_sql("SELECT rebal_date, ticker, lowvol FROM factors_raw", con)
    # trailing 60d vol = 1 / lowvol (raw factor), known at t
    vol_m = {d: 1.0 / g.set_index("ticker")["lowvol"] for d, g in raw.groupby("rebal_date")}

    start = preds.rebal_date.min()
    wsig = weekly_signals(con, close, start)
    wraw = F.compute_factor_panels(close, load_market(con, close.index))["lowvol"]
    vol_w = {d: 1.0 / wraw.loc[d].dropna() for d in wsig.rebal_date.unique()}

    base = BASE_COST_BPS + BASE_SLIP_BPS
    variants = {}
    for col in ["lgbm", "linear", "ew_prior"]:
        for side in ["ls", "lo"]:
            for scheme in ["ew", "vol"]:
                variants[f"{col}|{side}|{scheme}|monthly"] = build_weights(preds, col, side, scheme, vol_m)
    for side in ["ls", "lo"]:
        variants[f"ew_prior|{side}|ew|weekly"] = build_weights(wsig, "ew_prior", side, "ew", vol_w)

    # benchmark: equal-weight the whole universe, monthly (long-only, same costs)
    uni = pd.read_sql("SELECT rebal_date, ticker FROM universe WHERE rebal_date >= ?", con, params=[start])
    variants["universe|lo|ew|monthly"] = {d: pd.Series(1 / len(g), index=g.ticker.values)
                                          for d, g in uni.groupby("rebal_date")}

    daily, rows = {}, []
    for name, w in variants.items():
        bt = run(w, rets, base)
        bt["ret_gross"] = run(w, rets, 0.0)["ret"]
        daily[name] = bt
        rows.append(summary_row(name, bt, w))
    nifty = mkt.loc[daily["lgbm|ls|ew|monthly"].index]

    summary = pd.DataFrame(rows).set_index("variant")

    # cost sweep (net annual return and Sharpe vs cost), turnover is cost-independent
    sweep = []
    for name in ["lgbm|ls|ew|monthly", "linear|ls|ew|monthly", "ew_prior|ls|ew|monthly",
                 "ew_prior|ls|ew|weekly", "lgbm|lo|ew|monthly"]:
        for c in COST_GRID:
            bt = run(variants[name], rets, c)
            sweep.append({"variant": name, "cost_bps": c, **ann_stats(bt.ret)})
    sweep = pd.DataFrame(sweep)
    be = {}
    for name, g in sweep.groupby("variant"):
        be[name] = breakeven(g.cost_bps.values, g.cagr.values)
    summary["breakeven_cost_bps"] = pd.Series(be)

    RESULTS.mkdir(exist_ok=True)
    wide = pd.concat({k: v.ret for k, v in daily.items()}, axis=1)
    wide["nifty50"] = nifty
    wide.to_csv(RESULTS / "bt_daily.csv", index_label="date")
    summary.round(4).to_csv(RESULTS / "bt_summary.csv")
    sweep.round(5).to_csv(RESULTS / "bt_cost_sweep.csv", index=False)
    pd.set_option("display.width", 220)
    print(summary.round(3).to_string())
    print("\nNifty 50 (price index):", {k: round(v, 3) for k, v in ann_stats(nifty).items()})
    plot(wide, sweep, be)


def ann_stats(r, rf=0.0):
    r = r.dropna()
    yrs = len(r) / 252
    cagr = (1 + r).prod() ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252)
    ex = r - rf / 252
    return {"cagr": cagr, "vol": vol, "sharpe": ex.mean() / r.std() * np.sqrt(252)}


def breakeven(costs, cagr):
    """Cost (bps) at which net CAGR crosses zero, by linear interpolation."""
    if cagr[0] <= 0:
        return 0.0
    for i in range(1, len(costs)):
        if cagr[i] <= 0:
            return costs[i - 1] + (costs[i] - costs[i - 1]) * cagr[i - 1] / (cagr[i - 1] - cagr[i])
    return np.inf


def summary_row(name, bt, w):
    sig, side, scheme, freq = name.split("|")
    rf = RF_ANNUAL if side == "lo" else 0.0
    s = ann_stats(bt.ret, rf)
    g = ann_stats(bt.ret_gross, rf)
    trades = bt.turnover[bt.turnover > 0]
    per_year = len(trades) / (len(bt) / 252)
    return {"variant": name, "signal": sig, "side": side, "sizing": scheme, "rebalance": freq,
            "cagr_net": s["cagr"], "cagr_gross": g["cagr"], "vol": s["vol"],
            "sharpe_net": s["sharpe"], "sharpe_gross": g["sharpe"],
            "turnover_per_rebal": trades.iloc[1:].mean(), "turnover_annual": trades.iloc[1:].mean() * per_year,
            "cost_drag_annual": bt.cost.sum() / (len(bt) / 252), "start": bt.index[0], "end": bt.index[-1]}


def plot(wide, sweep, be):
    plt = style()
    # cost sensitivity, the headline chart
    fig, ax = plt.subplots(figsize=(8, 4.2))
    names = ["lgbm|ls|ew|monthly", "linear|ls|ew|monthly", "ew_prior|ls|ew|monthly", "ew_prior|ls|ew|weekly"]
    labels = ["LightGBM L/S monthly", "Linear L/S monthly", "EW composite L/S monthly", "EW composite L/S weekly"]
    for i, (n, lab) in enumerate(zip(names, labels)):
        g = sweep[sweep.variant == n]
        ax.plot(g.cost_bps, g.cagr * 100, color=SERIES[i], label=lab)
        b = be[n]
        if 0 < b < max(COST_GRID):
            ax.plot([b], [0], "o", ms=8, color=SERIES[i], mec="#fcfcfb", mew=2, zorder=5)
            ax.annotate(f"{b:.0f} bps", (b, 0), textcoords="offset points", xytext=(4, 8),
                        fontsize=8, color=INK_2)
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.axvline(BASE_COST_BPS + BASE_SLIP_BPS, color=INK_2, lw=0.8, alpha=0.5)
    ax.annotate("base case 20 bps", (BASE_COST_BPS + BASE_SLIP_BPS, ax.get_ylim()[1]),
                textcoords="offset points", xytext=(4, -12), fontsize=8, color=INK_2)
    ax.set_xlabel("Assumed one-way cost per unit turnover (bps)")
    ax.set_ylabel("Net CAGR, %")
    ax.set_title("Long-short performance vs assumed cost (dots = breakeven)")
    ax.legend(loc="upper right")
    fig.savefig(FIGURES / "cost_sensitivity.png"); plt.close(fig)

    # equity curves, net of base costs
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    idx = pd.to_datetime(wide.index)
    for i, (n, lab) in enumerate(zip(names[:3], labels[:3])):
        axes[0].plot(idx, (1 + wide[n]).cumprod(), color=SERIES[i], lw=1.6, label=lab)
    axes[0].set_title("Long-short, net of 20 bps")
    axes[0].set_ylabel("Growth of 1")
    lo = [("lgbm|lo|ew|monthly", "LightGBM top decile"), ("universe|lo|ew|monthly", "Universe equal-weight"),
          ("nifty50", "Nifty 50 (price index)")]
    for i, (n, lab) in enumerate(lo):
        axes[1].plot(idx, (1 + wide[n]).cumprod(), color=SERIES[i], lw=1.6, label=lab)
    axes[1].set_yscale("log")
    axes[1].set_title("Long-only, net of 20 bps (log scale)")
    for ax in axes:
        ax.legend(loc="upper left")
    fig.savefig(FIGURES / "equity_curves.png"); plt.close(fig)


if __name__ == "__main__":
    main()
