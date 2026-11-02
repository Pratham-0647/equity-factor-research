"""Phase 2: seven price/volume factors, computed point-in-time.

Every factor is a trailing-window statistic of data dated <= the rebalance
date t. Panels are built on the dense trading-day grid (trading_days), so
"21 days" means 21 trading days. Factors are then sampled at each monthly
rebalance date for the stocks in that date's liquidity universe.

Output table: factors_raw(rebal_date, ticker, sector, <7 factor columns>)

Factor definitions (sign: as specified in the build guide)
---------------------------------------------------------
mom_12_1   close[t-21] / close[t-252] - 1         (skip the last month)
strev      -(close[t] / close[t-21] - 1)          (short-term reversal)
lowvol     1 / std(daily returns, 60d)            (annualised vol, inverted)
idio_vol   std of residuals of r_i = a + b*r_mkt, 60d (annualised; raw, not inverted)
liquidity  log(universe.median_tv)                 (trailing 60d median close*volume,
                                                  taken from the SQL universe so the
                                                  universe and the factor share one
                                                  definition; log is monotone)
high_52w   close[t] / max(close, 252d)
beta       cov(r_i, r_mkt) / var(r_mkt), 252d

Market = Nifty 50 index (^NSEI). Returns are never forward-filled: a missing
price gives a missing return, and windows need MIN_FRAC of their days present.
Price *levels* used for lookback points (momentum, reversal) may be carried
forward at most FFILL_LIMIT days, so a one-day gap does not drop a stock.
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "factors.db"

ANN = np.sqrt(252)
MIN_FRAC = 0.8      # share of a window that must have data
FFILL_LIMIT = 5     # max trading days a price level can be carried forward

FACTORS = ["mom_12_1", "strev", "lowvol", "idio_vol", "liquidity", "high_52w", "beta"]


def load_panels(con):
    """Wide (date x ticker) close panel and market series on the trading-day grid."""
    days = pd.read_sql("SELECT date FROM trading_days ORDER BY day_idx", con)["date"]
    p = pd.read_sql("SELECT date, ticker, close FROM prices", con)
    close = p.pivot(index="date", columns="ticker", values="close").reindex(days)
    mkt = (pd.read_sql("SELECT date, close FROM nifty_index", con)
             .set_index("date")["close"].reindex(days))
    return close, mkt


def _minp(window):
    return int(np.ceil(MIN_FRAC * window))


def rolling_regression(r, m, window):
    """Rolling OLS of each column of r on m (with intercept), pairwise complete.

    Returns (beta, residual_std) panels. Implemented with rolling sums so it is
    vectorised across all tickers; each window ends at (and includes) row t.
    """
    m2 = pd.DataFrame(np.repeat(m.to_numpy()[:, None], r.shape[1], axis=1),
                      index=r.index, columns=r.columns)
    valid = r.notna() & m2.notna()
    x = m2.where(valid)
    y = r.where(valid)

    roll = lambda df: df.rolling(window, min_periods=1).sum()
    n = roll(valid.astype(float))
    sx, sy = roll(x.fillna(0)), roll(y.fillna(0))
    sxx, syy, sxy = roll((x * x).fillna(0)), roll((y * y).fillna(0)), roll((x * y).fillna(0))

    cov = sxy - sx * sy / n
    varx = sxx - sx * sx / n
    vary = syy - sy * sy / n
    beta = cov / varx
    ssr = (vary - beta * cov).clip(lower=0)            # residual sum of squares
    resid_std = np.sqrt(ssr / (n - 2))

    ok = n >= _minp(window)
    return beta.where(ok), resid_std.where(ok)


def compute_factor_panels(close, mkt, mom_lookback=252, skip=21, vol_window=60):
    """Six price-based factors as date x ticker panels. Row t uses rows <= t only.

    Liquidity comes from the SQL universe table (see sample_at_rebalance).
    mom_lookback / skip / vol_window default to the specification above and
    are varied only in the parameter-sensitivity analysis (Phase 7).
    """
    ret = close.pct_change(fill_method=None)
    mret = mkt.pct_change(fill_method=None)
    lvl = close.ffill(limit=FFILL_LIMIT)

    out = {}
    out["mom_12_1"] = lvl.shift(skip) / lvl.shift(mom_lookback) - 1
    out["strev"] = -(close / lvl.shift(21) - 1)

    vol60 = ret.rolling(vol_window, min_periods=_minp(vol_window)).std() * ANN
    out["lowvol"] = 1.0 / vol60.where(vol60 > 0)

    _, resid60 = rolling_regression(ret, mret, vol_window)
    out["idio_vol"] = resid60 * ANN

    hi = close.rolling(252, min_periods=_minp(252)).max()
    out["high_52w"] = close / hi

    beta252, _ = rolling_regression(ret, mret, 252)
    out["beta"] = beta252

    # A factor is only defined on a day the stock actually printed a price.
    return {k: v.where(close.notna()) for k, v in out.items()}


def sample_at_rebalance(panels, universe, sectors):
    """Stack panels, keep (rebal_date, ticker) pairs in the universe, add liquidity."""
    long = pd.concat({k: v.stack(future_stack=True) for k, v in panels.items()}, axis=1)
    long.index.names = ["rebal_date", "ticker"]
    long = long.reset_index()
    df = universe.merge(long, on=["rebal_date", "ticker"], how="left")
    df["liquidity"] = np.log(df["median_tv"].where(df["median_tv"] > 0))
    df = df.merge(sectors, on="ticker", how="left")
    df = df.replace([np.inf, -np.inf], np.nan)
    return df[["rebal_date", "ticker", "sector"] + FACTORS]


def build(con):
    close, mkt = load_panels(con)
    panels = compute_factor_panels(close, mkt)
    universe = pd.read_sql("SELECT rebal_date, ticker, median_tv FROM universe", con)
    sectors = pd.read_sql("SELECT ticker, sector FROM sectors", con)
    return sample_at_rebalance(panels, universe, sectors)


def main():
    con = sqlite3.connect(DB)
    df = build(con)
    df.to_sql("factors_raw", con, if_exists="replace", index=False)
    con.execute("CREATE INDEX IF NOT EXISTS ix_factors_raw ON factors_raw(rebal_date, ticker)")
    con.commit()

    print(f"factors_raw: {len(df):,} rows, {df.rebal_date.nunique()} rebalance dates")
    print("\nCoverage (share non-null):")
    print(df[FACTORS].notna().mean().round(3).to_string())
    print("\nCross-sectional summary (pooled):")
    print(df[FACTORS].describe(percentiles=[.01, .5, .99]).T.round(3).to_string())
    con.close()


if __name__ == "__main__":
    main()
