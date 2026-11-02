"""Shared helpers: paths, forward returns, factor priors, plot style."""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "factors.db"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

FACTORS = ["mom_12_1", "strev", "lowvol", "idio_vol", "liquidity", "high_52w", "beta"]

# Expected sign of each factor's relation to future returns, fixed from the
# published literature BEFORE any IC was computed on this data (committed with
# Phase 4, ahead of the results). Used to orient composites so that no sign is
# fitted to the sample.
#   momentum (Jegadeesh-Titman), reversal (Jegadeesh 1990; strev is already
#   negated), low vol / low idio vol (Ang et al. 2006), illiquidity premium
#   (Amihud 2002), 52-week high (George-Hwang 2004), betting against beta
#   (Frazzini-Pedersen 2014).
PRIOR_SIGN = {"mom_12_1": 1, "strev": 1, "lowvol": 1, "idio_vol": -1,
              "liquidity": -1, "high_52w": 1, "beta": -1}

ENTRY_LAG = 1       # signal at close t, trade at close t+1
FFILL_LIMIT = 5


def connect():
    return sqlite3.connect(DB)


def load_close(con):
    days = pd.read_sql("SELECT date FROM trading_days ORDER BY day_idx", con)["date"]
    p = pd.read_sql("SELECT date, ticker, close FROM prices", con)
    return p.pivot(index="date", columns="ticker", values="close").reindex(days)


def load_market(con, index):
    s = pd.read_sql("SELECT date, close FROM nifty_index", con).set_index("date")["close"]
    return s.reindex(index)


def forward_returns(close, dates, horizon, lag=ENTRY_LAG):
    """Return from close[t+lag] to close[t+lag+horizon] for each t in dates.

    Long format (rebal_date, ticker, fwd). Entry/exit price levels may be carried
    forward at most FFILL_LIMIT days; beyond that the return is missing. If the
    exit falls past the end of the data the return is missing (never truncated).
    """
    lvl = close.ffill(limit=FFILL_LIMIT)
    pos = close.index.get_indexer(pd.Index(dates))
    if (pos < 0).any():
        raise ValueError("rebalance date not on trading-day grid")
    n = len(close)
    rows = []
    for d, i in zip(dates, pos):
        a, b = i + lag, i + lag + horizon
        if b >= n:
            continue
        r = lvl.iloc[b] / lvl.iloc[a] - 1
        rows.append(pd.DataFrame({"rebal_date": d, "ticker": r.index, "fwd": r.values}))
    out = pd.concat(rows, ignore_index=True)
    return out.dropna()


def newey_west_t(x, lags):
    """t-statistic of the mean of x with Bartlett-kernel HAC standard error."""
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = len(x)
    e = x - x.mean()
    s = e @ e / n
    for k in range(1, lags + 1):
        s += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return x.mean() / np.sqrt(s / n)


# ---------------------------------------------------------------- plot style
# Reference palette from the dataviz skill (light mode, static PNG output).
SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
GRID = "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DIVERGING = ["#2a78d6", "#f0efec", "#e34948"]   # blue <-> gray <-> red
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8, "axes.labelcolor": INK_2,
        "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "semibold",
        "axes.labelsize": 9, "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": matplotlib.cycler(color=SERIES), "lines.linewidth": 2,
        "legend.frameon": False, "legend.fontsize": 8, "font.size": 9,
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    })
    return plt


def diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("div", DIVERGING)


def sequential_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("seq", SEQ_BLUE)
