"""Phase 3: cross-sectional processing of factors_raw -> factors_z.

At EACH rebalance date, independently, for each factor:
  1. winsorize at that date's 1st / 99th percentile
  2. z-score with that date's cross-sectional mean and std
  3. demean within sector (subtract that date's sector mean of the z-scores)

No statistic is pooled across dates, so a value at date t depends only on the
cross-section at t (which itself uses only data <= t, see factors.py).

Missing values stay missing: they are excluded from the date's statistics and
are not imputed. Sectors with fewer than MIN_SECTOR_SIZE names with a value on
a date are pooled into one "Other" group for step 3; otherwise a one-stock
sector (Textiles has one name) is always demeaned to exactly zero.

Sector labels are today's NSE industry classification (static), a mild
look-ahead that the free data cannot avoid.
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from factors import FACTORS

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "factors.db"

LOWER_Q, UPPER_Q = 0.01, 0.99
MIN_SECTOR_SIZE = 3


def winsorize(x, lower=LOWER_Q, upper=UPPER_Q):
    lo, hi = x.quantile([lower, upper])
    return x.clip(lo, hi)


def zscore(x):
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else x * np.nan


def sector_demean(x, sector, min_size=MIN_SECTOR_SIZE):
    counts = x.groupby(sector).transform("count")
    grp = sector.where(counts >= min_size, "Other")
    return x - x.groupby(grp).transform("mean")


def process_cross_section(df, factors=FACTORS):
    """One rebalance date: winsorize -> z-score -> sector demean, per factor."""
    out = df[["rebal_date", "ticker", "sector"]].copy()
    sector = df["sector"].fillna("Unknown")
    for f in factors:
        x = winsorize(df[f])
        x = zscore(x)
        out[f] = sector_demean(x, sector)
    return out


def process(raw, factors=FACTORS):
    parts = [process_cross_section(g, factors) for _, g in raw.groupby("rebal_date", sort=True)]
    return pd.concat(parts, ignore_index=True)


def main():
    con = sqlite3.connect(DB)
    raw = pd.read_sql("SELECT * FROM factors_raw", con)
    z = process(raw)
    z.to_sql("factors_z", con, if_exists="replace", index=False)
    con.execute("CREATE INDEX IF NOT EXISTS ix_factors_z ON factors_z(rebal_date, ticker)")
    con.commit()

    print(f"factors_z: {len(z):,} rows, {z.rebal_date.nunique()} rebalance dates")
    by_date = z.groupby("rebal_date")[FACTORS]
    print("\nPer-date mean (should be ~0), max |mean| across dates:")
    print(by_date.mean().abs().max().round(6).to_string())
    print("\nPer-date std, median across dates:")
    print(by_date.std().median().round(3).to_string())
    sect = z.groupby(["rebal_date", "sector"])[FACTORS].transform("count")
    print(f"\nMax |sector mean| among sectors with >= {MIN_SECTOR_SIZE} names:",
          z[FACTORS].where(sect >= MIN_SECTOR_SIZE)
           .groupby([z.rebal_date, z.sector]).mean().abs().max().max().round(10))
    print("\nPooled correlation of processed factors:")
    print(z[FACTORS].corr().round(2).to_string())
    con.close()


if __name__ == "__main__":
    main()
