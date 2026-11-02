-- Point-in-time liquidity universe (monthly rebalance).
--
-- For each month-end rebalance date t, a stock is eligible if it has at least
-- :min_history trading days of price history up to and including t, and a full
-- 60-day window of traded value. Eligible stocks are ranked by TRAILING 60-day
-- MEDIAN traded value (close * volume); the top :top_n form the universe.
--
-- Only data dated <= t is used, so the universe is causal. No historical
-- index-membership file is needed.
--
-- Caveat: the candidate ticker list is today's Nifty 200, so stocks that were
-- delisted or dropped before today are absent. The liquidity rule fixes
-- *selection-by-recent-performance* bias within the list; it does not recover
-- missing delisted names.

DROP TABLE IF EXISTS rebalance_dates;
CREATE TABLE rebalance_dates AS
SELECT MAX(date) AS date
FROM trading_days
GROUP BY substr(date, 1, 7);

DROP TABLE IF EXISTS tv;
CREATE TABLE tv AS
SELECT p.ticker, p.date, d.day_idx,
       p.close * p.volume AS traded_value
FROM prices p
JOIN trading_days d ON d.date = p.date;
CREATE INDEX ix_tv ON tv(ticker, day_idx);

-- trailing 60-day window rows at each rebalance date
DROP TABLE IF EXISTS tv_window;
CREATE TABLE tv_window AS
SELECT r.date AS rebal_date, w.ticker, w.traded_value,
       ROW_NUMBER() OVER (PARTITION BY r.date, w.ticker ORDER BY w.traded_value) AS rn,
       COUNT(*)     OVER (PARTITION BY r.date, w.ticker)                        AS n
FROM rebalance_dates r
JOIN trading_days rd ON rd.date = r.date
JOIN tv w ON w.day_idx BETWEEN rd.day_idx - 59 AND rd.day_idx;

DROP TABLE IF EXISTS liq_median;
CREATE TABLE liq_median AS
SELECT rebal_date, ticker, n AS n_obs, AVG(traded_value) AS median_tv
FROM tv_window
WHERE n = 60                                  -- full window only
  AND rn IN ((n + 1) / 2, (n + 2) / 2)        -- middle one or two rows
GROUP BY rebal_date, ticker, n;

DROP TABLE IF EXISTS history_count;
CREATE TABLE history_count AS
SELECT r.date AS rebal_date, p.ticker, COUNT(*) AS n_days
FROM rebalance_dates r
JOIN prices p ON p.date <= r.date
GROUP BY r.date, p.ticker;

DROP TABLE IF EXISTS universe;
CREATE TABLE universe AS
SELECT rebal_date, ticker, median_tv, liq_rank
FROM (
  SELECT l.rebal_date, l.ticker, l.median_tv,
         ROW_NUMBER() OVER (PARTITION BY l.rebal_date ORDER BY l.median_tv DESC) AS liq_rank
  FROM liq_median l
  JOIN history_count h ON h.rebal_date = l.rebal_date AND h.ticker = l.ticker
  WHERE h.n_days >= 252
    AND l.median_tv > 0
)
WHERE liq_rank <= 200;
CREATE INDEX ix_universe ON universe(rebal_date, ticker);
