"""Load prices.parquet + sectors.csv into SQLite (data/factors.db).

Tables
------
prices(date, ticker, open, high, low, close, volume)   -- long format, adjusted OHLC
sectors(ticker, sector)
trading_days(date, day_idx)                            -- dense trading-day index
corporate_actions(...)                                 -- manual fixes applied (see below)

Corporate actions: yfinance leaves some bonuses and demergers unadjusted, which
shows up as a one-day -35% to -75% "return" that no holder experienced.
corporate_actions.csv lists events reviewed by hand (|return| > 25% on a day the
index moved < 3%). For action=adjust, all OHLC before the ex-date is multiplied
by `factor` and volume divided by it (traded value unchanged); factor=observed
uses that day's close ratio, i.e. sets the ex-date return to zero, a standard
approximation for demergers. action=truncate drops all history before the date.
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "factors.db"


def apply_corporate_actions(prices, ca):
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    ca = ca.copy()
    applied = []
    for i, a in ca.iterrows():
        tk = prices["ticker"] == a.ticker
        before = tk & (prices["date"] < a.date)
        on = tk & (prices["date"] == a.date)
        if not on.any() or not before.any():
            raise ValueError(f"corporate action {a.ticker} {a.date}: no price on/before date")
        if a.action == "truncate":
            prices = prices[~before].reset_index(drop=True)
            applied.append(np.nan)
            continue
        prev_close = prices.loc[before, "close"].iloc[-1]
        f = (prices.loc[on, "close"].iloc[0] / prev_close
             if str(a.factor) == "observed" else float(a.factor))
        prices.loc[before, ["open", "high", "low", "close"]] *= f
        prices.loc[before, "volume"] /= f
        applied.append(f)
    ca["factor_applied"] = applied
    return prices, ca


def main():
    DB.parent.mkdir(exist_ok=True)
    px = pd.read_parquet(ROOT / "prices.parquet")

    frames = []
    for tkr in px.columns.get_level_values(0).unique():
        if tkr == "^NSEI":
            continue
        d = px[tkr].dropna(subset=["Close"]).copy()
        d = d[d["Close"] > 0]
        d["ticker"] = tkr.replace(".NS", "")
        d.index.name = "date"
        frames.append(d.reset_index())
    prices = pd.concat(frames, ignore_index=True)
    prices.columns = [c.lower() for c in prices.columns]
    prices = prices[["date", "ticker", "open", "high", "low", "close", "volume"]]
    prices["date"] = prices["date"].dt.strftime("%Y-%m-%d")
    prices["volume"] = prices["volume"].fillna(0)
    prices, ca = apply_corporate_actions(prices, pd.read_csv(ROOT / "corporate_actions.csv"))

    idx = px["^NSEI"]["Close"].dropna()
    index_df = pd.DataFrame({"date": idx.index.strftime("%Y-%m-%d"), "close": idx.values})

    sec = pd.read_csv(ROOT / "sectors.csv").rename(columns={"Symbol": "ticker", "Industry": "sector"})

    days = sorted(prices["date"].unique())
    trading_days = pd.DataFrame({"date": days, "day_idx": range(len(days))})

    if DB.exists():
        DB.unlink()
    con = sqlite3.connect(DB)
    prices.to_sql("prices", con, index=False)
    index_df.to_sql("nifty_index", con, index=False)
    sec.to_sql("sectors", con, index=False)
    trading_days.to_sql("trading_days", con, index=False)
    ca.to_sql("corporate_actions", con, index=False)
    con.execute("CREATE INDEX ix_prices ON prices(ticker, date)")
    con.execute("CREATE UNIQUE INDEX ix_days ON trading_days(date)")
    con.commit()

    for t in ("prices", "nifty_index", "sectors", "trading_days"):
        n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"{t}: {n:,} rows")
    con.close()


if __name__ == "__main__":
    main()
