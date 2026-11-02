"""Phase 8a: export tidy tables for the Power BI dashboard (powerbi/data/*.csv).

Star schema:
  dim_strategy   one row per backtest variant (signal, side, sizing, rebalance)
  dim_date       calendar for the backtest period
  fact_daily     date x strategy: daily net return, NAV, drawdown
  fact_monthly   month x strategy: monthly net return
  fact_cost      strategy x cost_bps: net CAGR / Sharpe (cost sensitivity)
  fact_metrics   strategy: CAGR, Sharpe, Sortino, max DD, Calmar, turnover, beta
  fact_ic        month x factor: rank IC (21d)
  fact_deciles   factor x decile: mean 21d forward return
  fact_sens      lookback grid and portfolio grid results
"""
import pandas as pd

from common import FACTORS, RESULTS, ROOT

OUT = ROOT / "powerbi" / "data"

LABELS = {"lgbm": "LightGBM", "linear": "Linear", "ew_prior": "EW composite",
          "universe": "Universe EW", "nifty50": "Nifty 50"}


def strategy_dim(cols):
    rows = []
    for c in cols:
        if c == "nifty50":
            rows.append({"strategy": c, "signal": "Nifty 50", "side": "benchmark", "sizing": "",
                         "rebalance": "", "label": "Nifty 50 (price index)"})
            continue
        sig, side, sizing, freq = c.split("|")
        rows.append({"strategy": c, "signal": LABELS[sig],
                     "side": {"ls": "Long-short", "lo": "Long-only"}[side],
                     "sizing": {"ew": "Equal weight", "vol": "Inverse vol"}[sizing], "rebalance": freq,
                     "label": f"{LABELS[sig]} {'L/S' if side == 'ls' else 'L/O'} {sizing} {freq}"})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(RESULTS / "bt_daily.csv", index_col="date")
    strategy_dim(daily.columns).to_csv(OUT / "dim_strategy.csv", index=False)

    d = pd.to_datetime(daily.index)
    pd.DataFrame({"date": d, "year": d.year, "month": d.month, "month_name": d.strftime("%b"),
                  "year_month": d.strftime("%Y-%m")}).to_csv(OUT / "dim_date.csv", index=False)

    long = daily.reset_index().melt(id_vars="date", var_name="strategy", value_name="ret")
    long["nav"] = long.groupby("strategy")["ret"].transform(lambda r: (1 + r).cumprod())
    long["drawdown"] = long["nav"] / long.groupby("strategy")["nav"].cummax() - 1
    long.to_csv(OUT / "fact_daily.csv", index=False)

    long["year_month"] = long["date"].str[:7]
    monthly = long.groupby(["strategy", "year_month"])["ret"].apply(lambda r: (1 + r).prod() - 1).reset_index()
    monthly.to_csv(OUT / "fact_monthly.csv", index=False)

    pd.read_csv(RESULTS / "bt_cost_sweep.csv").to_csv(OUT / "fact_cost.csv", index=False)
    pd.read_csv(RESULTS / "risk_metrics.csv").rename(columns={"variant": "strategy"}).to_csv(
        OUT / "fact_metrics.csv", index=False)

    ic = pd.read_csv(RESULTS / "ic_timeseries.csv").melt(id_vars="rebal_date", var_name="factor", value_name="ic")
    ic.to_csv(OUT / "fact_ic.csv", index=False)
    dec = pd.read_csv(RESULTS / "deciles.csv").melt(id_vars="factor", var_name="decile", value_name="mean_fwd_21d")
    dec.to_csv(OUT / "fact_deciles.csv", index=False)

    a = pd.read_csv(RESULTS / "sens_lookback.csv")
    a.to_csv(OUT / "fact_sens_lookback.csv", index=False)
    pd.read_csv(RESULTS / "sens_portfolio.csv").to_csv(OUT / "fact_sens_portfolio.csv", index=False)

    for f in sorted(OUT.glob("*.csv")):
        print(f"{f.name}: {sum(1 for _ in open(f)) - 1:,} rows")


if __name__ == "__main__":
    main()
