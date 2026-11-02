# Power BI dashboard

`.pbix` files can only be authored in Power BI Desktop (Windows), so this folder
holds everything up to that point: the data model as tidy CSVs, the DAX
measures, and a page-by-page build spec. Building the file takes about 30
minutes in Power BI Desktop.

Regenerate the data with `python src/export_powerbi.py` (after the pipeline has run).

## 1. Load the data

Home → Get data → Text/CSV → load every file in `powerbi/data/`. In Power Query
set `date` columns to Date and returns to Decimal Number.

| Table | Grain | Key columns |
|---|---|---|
| `dim_strategy` | one row per strategy | `strategy`, `signal`, `side`, `sizing`, `rebalance`, `label` |
| `dim_date` | one row per trading day | `date`, `year`, `month`, `month_name`, `year_month` |
| `fact_daily` | date × strategy | `date`, `strategy`, `ret`, `nav`, `drawdown` |
| `fact_monthly` | month × strategy | `strategy`, `year_month`, `ret` |
| `fact_metrics` | strategy | CAGR, Sharpe, Sortino, max DD, Calmar, turnover, beta |
| `fact_cost` | strategy × cost | `variant`, `cost_bps`, `cagr`, `sharpe` |
| `fact_ic` | month × factor | `rebal_date`, `factor`, `ic` |
| `fact_deciles` | factor × decile | `factor`, `decile`, `mean_fwd_21d` |
| `fact_sens_lookback`, `fact_sens_portfolio` | grid cells | parameters + net Sharpe |

## 2. Relationships (Model view)

- `dim_strategy[strategy]` 1 → * `fact_daily[strategy]`, `fact_monthly[strategy]`, `fact_metrics[strategy]`
- `dim_strategy[strategy]` 1 → * `fact_cost[variant]`
- `dim_date[date]` 1 → * `fact_daily[date]`
- Mark `dim_date` as the date table.

## 3. DAX measures (on `fact_daily`)

```dax
Days = COUNTROWS ( fact_daily )
Years = [Days] / 252

Growth = PRODUCTX ( fact_daily, 1 + fact_daily[ret] )
CAGR = POWER ( [Growth], 1 / [Years] ) - 1

Ann Vol = STDEVX.S ( fact_daily, fact_daily[ret] ) * SQRT ( 252 )

-- long-short books are self-financing: rf = 0; for long-only use 0.06
RF Annual = IF ( SELECTEDVALUE ( dim_strategy[side] ) = "Long-short", 0, 0.06 )
Sharpe =
    DIVIDE ( AVERAGEX ( fact_daily, fact_daily[ret] - [RF Annual] / 252 ),
             STDEVX.S ( fact_daily, fact_daily[ret] ) ) * SQRT ( 252 )

Downside Dev =
    SQRT ( AVERAGEX ( fact_daily,
        POWER ( MIN ( fact_daily[ret] - [RF Annual] / 252, 0 ), 2 ) ) )
Sortino =
    DIVIDE ( AVERAGEX ( fact_daily, fact_daily[ret] - [RF Annual] / 252 ), [Downside Dev] ) * SQRT ( 252 )

Max Drawdown = MIN ( fact_daily[drawdown] )
Calmar = DIVIDE ( [CAGR], ABS ( [Max Drawdown] ) )

Monthly Return = SUM ( fact_monthly[ret] )   -- one row per strategy-month
```

Max Drawdown reads the precomputed column, which is measured over the full
period. If you add a date slicer, recompute drawdown in DAX or treat it as full-period.

## 4. Pages

1. **Overview**: strategy slicer (`dim_strategy[label]`). Cards: CAGR, Sharpe,
   Sortino, Max Drawdown, Calmar. Line chart: `fact_daily[nav]` by date, legend =
   label (pick at most 4 strategies; log scale for long-only).
2. **Drawdown**: area chart of `fact_daily[drawdown]` by date for the selected strategy.
3. **Monthly heatmap**: matrix, rows = `dim_date[year]`, columns = `dim_date[month_name]`,
   values = `Monthly Return`; conditional formatting with a diverging scale
   (red negative, grey 0, blue positive).
4. **Cost sensitivity** (headline): line chart, x = `fact_cost[cost_bps]`,
   y = `fact_cost[cagr]`, legend = `variant`; constant line at 0 and at 20 bps.
   Breakevens: LightGBM ≈ 39 bps, Linear ≈ 56 bps.
5. **Signals**: column chart of mean `fact_ic[ic]` by factor; line chart of the
   running sum of IC by month; clustered columns of `fact_deciles` (x = decile, small multiples by factor).
6. **Robustness**: matrices of the two sensitivity grids with the same diverging scale.

Use one colour per strategy and keep it fixed across pages
(LightGBM `#2a78d6`, Linear `#eb6834`, EW composite `#1baf7a`, benchmarks `#eda100`).
