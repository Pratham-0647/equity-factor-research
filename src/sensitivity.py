"""Phase 7b: parameter sensitivity.

Grid A - factor lookbacks (full pipeline rerun per cell: factors -> per-date
processing -> purged walk-forward LightGBM/linear -> backtest):
    momentum formation window  MOM_GRID trading days (skip 21)
    volatility / idio-vol window VOL_GRID trading days
Grid B - portfolio construction for the headline LightGBM signal (no refit):
    selection quantile Q_GRID x one-way cost COST_GRID bps

Metric: net Sharpe of the monthly long-short book, plus out-of-sample IC.
The default specification (252, 60) is one cell of grid A; a robust signal
should not depend on landing on it.
Outputs: results/sens_lookback.csv, results/sens_portfolio.csv,
figures/sensitivity_lookback.png, figures/sensitivity_portfolio.png
"""
import sys

import numpy as np
import pandas as pd

import backtest as B
import factors as F
import model as M
import preprocess as P
from common import FIGURES, INK_2, RESULTS, connect, load_close, load_market, sequential_cmap, diverging_cmap, style

MOM_GRID = [126, 189, 252, 378]
VOL_GRID = [20, 40, 60, 120]
Q_GRID = [0.05, 0.10, 0.20, 0.30]
COST_GRID = [10, 20, 30, 40]
BASE = B.BASE_COST_BPS + B.BASE_SLIP_BPS


def lookback_cell(con, close, mkt, rets, uni, sectors, mom, vol):
    panels = F.compute_factor_panels(close, mkt, mom_lookback=mom, vol_window=vol)
    z = P.process(F.sample_at_rebalance(panels, uni, sectors))
    df = M.load_panel(con, z=z, close=close)
    preds, _ = M.walk_forward(df)
    sc = M.score(preds)
    out = {"mom_lookback": mom, "vol_window": vol}
    for col in ["lgbm", "linear"]:
        bt = B.run(B.build_weights(preds, col, "ls", "ew", {}), rets, BASE)
        st = B.ann_stats(bt.ret)
        out.update({f"{col}_oos_ic": sc.loc[col, "oos_mean_ic"], f"{col}_sharpe_net": st["sharpe"],
                    f"{col}_cagr_net": st["cagr"]})
    return out


def grid_a():
    con = connect()
    close = load_close(con)
    mkt = load_market(con, close.index)
    rets = B.daily_returns(close)
    uni = pd.read_sql("SELECT rebal_date, ticker, median_tv FROM universe", con)
    sectors = pd.read_sql("SELECT ticker, sector FROM sectors", con)
    rows = []
    for mom in MOM_GRID:
        for vol in VOL_GRID:
            r = lookback_cell(con, close, mkt, rets, uni, sectors, mom, vol)
            print(r, flush=True)
            rows.append(r)
    pd.DataFrame(rows).round(4).to_csv(RESULTS / "sens_lookback.csv", index=False)


def grid_b():
    con = connect()
    close = load_close(con)
    rets = B.daily_returns(close)
    preds = pd.read_sql("SELECT * FROM predictions", con)
    rows = []
    for q in Q_GRID:
        w = {d: B.target_weights(g, "lgbm", "ls", "ew", q=q) for d, g in preds.groupby("rebal_date")}
        for c in COST_GRID:
            st = B.ann_stats(B.run(w, rets, c).ret)
            rows.append({"quantile": q, "cost_bps": c, "sharpe_net": st["sharpe"], "cagr_net": st["cagr"]})
    pd.DataFrame(rows).round(4).to_csv(RESULTS / "sens_portfolio.csv", index=False)


def heat(ax, m, xlabels, ylabels, xlab, ylab, title, mark=None, vmax=None):
    v = vmax or np.nanmax(np.abs(m.values))
    im = ax.imshow(m.values, cmap=diverging_cmap(), vmin=-v, vmax=v, origin="lower")
    ax.set_xticks(range(len(xlabels)), xlabels)
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.set_title(title); ax.grid(False)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m.values[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="#0b0b0b", fontweight="bold" if mark == (i, j) else "normal")
    if mark:
        ax.add_patch(__import__("matplotlib").patches.Rectangle((mark[1] - .5, mark[0] - .5), 1, 1,
                     fill=False, ec="#0b0b0b", lw=1.5))
    return im


def plot():
    plt = style()
    a = pd.read_csv(RESULTS / "sens_lookback.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, col, t in zip(axes, ["lgbm_sharpe_net", "linear_sharpe_net"], ["LightGBM", "Linear"]):
        m = a.pivot(index="mom_lookback", columns="vol_window", values=col)
        im = heat(ax, m, VOL_GRID, MOM_GRID, "Vol / idio-vol window (days)", "Momentum lookback (days)",
                  f"{t} L/S net Sharpe (20 bps)", mark=(MOM_GRID.index(252), VOL_GRID.index(60)), vmax=1.2)
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.savefig(FIGURES / "sensitivity_lookback.png"); plt.close(fig)

    b = pd.read_csv(RESULTS / "sens_portfolio.csv")
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    m = b.pivot(index="quantile", columns="cost_bps", values="sharpe_net")
    im = heat(ax, m, COST_GRID, [f"{q:.0%}" for q in Q_GRID], "One-way cost (bps)", "Long/short quantile",
              "LightGBM L/S net Sharpe", mark=(Q_GRID.index(0.10), COST_GRID.index(20)), vmax=1.2)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(FIGURES / "sensitivity_portfolio.png"); plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("a", "all"):
        grid_a()
    if which in ("b", "all"):
        grid_b()
    if which in ("plot", "all"):
        plot()
