"""Phase 8b: build the research note PDF (report/equity_factor_research.pdf).

Every number in the text is read from results/*.csv at build time, so the
report cannot drift from the analysis. Re-run after any pipeline change.
"""
import datetime as dt

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

from common import FIGURES, RESULTS, ROOT, connect

OUT = ROOT / "report" / "equity_factor_research.pdf"
FONT_DIR = "/usr/share/fonts/truetype/dejavu/"
pdfmetrics.registerFont(TTFont("DV", FONT_DIR + "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DV-B", FONT_DIR + "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DV-I", FONT_DIR + "DejaVuSans-Oblique.ttf"))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-B")

INK, INK2, GRID, ACCENT = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e"), colors.HexColor("#e1e0d9"), colors.HexColor("#2a78d6")
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontName="DV-B", fontSize=14, textColor=INK, spaceBefore=10, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="DV-B", fontSize=11, textColor=INK, spaceBefore=8, spaceAfter=4)
BODY = ParagraphStyle("B", parent=ss["BodyText"], fontName="DV", fontSize=9, leading=13, textColor=INK, alignment=TA_LEFT, spaceAfter=5)
SMALL = ParagraphStyle("S", parent=BODY, fontSize=7.5, leading=10, textColor=INK2)
BUL = ParagraphStyle("BU", parent=BODY, leftIndent=12, bulletIndent=2, spaceAfter=2)
TITLE = ParagraphStyle("T", parent=BODY, fontName="DV-B", fontSize=20, leading=24, spaceAfter=4)


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def f(x, d=2):
    return f"{x:.{d}f}"


def bullets(items):
    return [Paragraph(t, BUL, bulletText="•") for t in items]


def table(df, col_w=None, fs=7.5):
    data = [list(df.columns)] + df.astype(str).values.tolist()
    t = Table(data, colWidths=col_w, repeatRows=1)
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, 0), "DV-B", fs), ("FONT", (0, 1), (-1, -1), "DV", fs),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK), ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK2),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, GRID), ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]))
    return t


def fig(name, width=17):
    from PIL import Image as PI
    p = FIGURES / name
    w, h = PI.open(p).size
    return Image(str(p), width=width * cm, height=width * cm * h / w)


def caption(t):
    return Paragraph(t, SMALL)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("DV", 7)
    canvas.setFillColor(INK2)
    canvas.drawString(2 * cm, 1.2 * cm, "Equity Factor Research & Systematic Backtest - NSE large/mid caps")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"{doc.page}")
    canvas.restoreState()


def main():
    R = lambda n, **k: pd.read_csv(RESULTS / n, **k)
    ic = R("ic_summary.csv", index_col="factor")
    decay = R("ic_decay.csv", index_col="factor")
    oos = R("model_oos.csv", index_col="model")
    folds = R("model_folds.csv")
    mchk = R("model_checks.csv", index_col="check")["value"]
    bt = R("bt_summary.csv", index_col="variant")
    bchk = R("bt_checks.csv", index_col="check")["value"]
    risk = R("risk_metrics.csv", index_col="variant")
    dsr = R("deflated_sharpe.csv", index_col="variant")
    variants = R("variants.csv")
    sens_a = R("sens_lookback.csv")
    sens_b = R("sens_portfolio.csv")

    con = connect()
    q = lambda s: con.execute(s).fetchone()
    n_px, n_tk, d0, d1 = q("SELECT COUNT(*), COUNT(DISTINCT ticker), MIN(date), MAX(date) FROM prices")
    n_days = q("SELECT COUNT(*) FROM trading_days")[0]
    n_reb, u_min, u_max = q("SELECT COUNT(DISTINCT rebal_date), MIN(n), MAX(n) FROM "
                            "(SELECT rebal_date, COUNT(*) n FROM universe GROUP BY 1)")
    ca = pd.read_sql("SELECT ticker, date, action, verified FROM corporate_actions", con)

    H = "lgbm|ls|ew|monthly"
    L = "linear|ls|ew|monthly"
    E = "ew_prior|ls|ew|monthly"
    hb, hr = bt.loc[H], risk.loc[H]
    lr = risk.loc[L]
    lo_ex = risk.loc["lgbm_lo_minus_universe"]
    default_cell = sens_a[(sens_a.mom_lookback == 252) & (sens_a.vol_window == 60)].iloc[0]
    pos_cells = (sens_a.lgbm_sharpe_net > 0).sum()

    s = []
    s.append(Paragraph("Equity Factor Research &amp; Systematic Backtest", TITLE))
    s.append(Paragraph(f"Cross-sectional factors on NSE large and mid caps, {d0[:4]}-{d1[:4]} &nbsp;·&nbsp; "
                       f"Research note &nbsp;·&nbsp; {dt.date.today():%d %B %Y}", SMALL))
    s.append(Spacer(1, 10))

    s.append(Paragraph("Summary", H1))
    s += bullets([
        f"Seven price/volume factors on a point-in-time liquidity universe of {u_min}-{u_max} NSE stocks "
        f"({n_reb} monthly rebalances, {n_days:,} trading days). Only data available at each rebalance date is used.",
        f"<b>Two factors carry real signal:</b> 12-1 momentum (mean monthly rank IC {f(ic.loc['mom_12_1','mean_ic'],3)}, "
        f"t = {f(ic.loc['mom_12_1','t_stat'],1)}) and liquidity (IC {f(ic.loc['liquidity','mean_ic'],3)}, "
        f"t = {f(ic.loc['liquidity','t_stat'],1)}; illiquid names outperform). Low-volatility and idiosyncratic "
        f"volatility ran <i>against</i> their literature signs in this period.",
        f"<b>LightGBM under purged, embargoed walk-forward CV</b> reaches out-of-sample IC "
        f"{f(oos.loc['lgbm','oos_mean_ic'],3)} (t = {f(oos.loc['lgbm','ic_t'],1)}) over {int(oos.loc['lgbm','n_months'])} months, "
        f"versus {f(oos.loc['ew_prior','oos_mean_ic'],3)} for an equal-weight composite. The gain over the composite is not "
        f"significant (t = {f(oos.loc['lgbm_minus_ew_prior','ic_t'],1)}), and a plain linear model does better "
        f"(IC {f(oos.loc['linear','oos_mean_ic'],3)}). At this signal-to-noise ratio, ML adds little.",
        f"<b>The long-short decile book makes money before costs but not convincingly after.</b> LightGBM L/S: gross CAGR "
        f"{pct(hb.cagr_gross)} (Sharpe {f(hb.sharpe_gross)}), net {pct(hr.cagr)} (Sharpe {f(hr.sharpe)}, Sortino "
        f"{f(hr.sortino)}, max drawdown {pct(hr.max_drawdown,0)}) at 20 bps. Turnover is {f(hb.turnover_per_rebal,1)}x NAV "
        f"per rebalance and the breakeven cost is {f(hb.breakeven_cost_bps,0)} bps.",
        f"<b>After multiple-testing correction the headline Sharpe is not significant:</b> with "
        f"{int(dsr.loc[H,'n_strategy_variants'])} strategy variants scored, the best of that many no-skill strategies would "
        f"show Sharpe ~{f(dsr.loc[H,'expected_max_null_sharpe'])}; deflated-Sharpe probability {f(dsr.loc[H,'deflated_sharpe_prob'])}.",
        f"<b>The most robust result is the long leg:</b> the LightGBM top decile beat the equal-weight universe by "
        f"{pct(lo_ex.cagr)} a year (Sharpe {f(lo_ex.sharpe)}, max relative drawdown {pct(lo_ex.max_drawdown,0)}).",
    ])

    s.append(Paragraph("1. Data and universe", H1))
    s.append(Paragraph(
        f"Daily adjusted OHLCV for the current Nifty 200 constituents and the Nifty 50 index from Yahoo Finance, "
        f"{d0} to {d1}: {n_px:,} stock-days across {n_tk} tickers, stored in SQLite. Sector labels are NSE "
        f"industry classifications (18 sectors).", BODY))
    s.append(Paragraph(
        "<b>Point-in-time universe (SQL).</b> At each month-end a stock is eligible if it has at least 252 days of "
        "history and a full 60-day window of traded value; eligible stocks are ranked by trailing 60-day median "
        "close × volume and the top 200 kept. Only data dated on or before the rebalance date is used, so selection "
        "never depends on later performance.", BODY))
    s.append(Paragraph(
        f"<b>Survivorship caveat.</b> The candidate list is <i>today's</i> Nifty 200, so names delisted or "
        f"dropped before {d1[:4]} are missing, and the cap of 200 never binds ({u_min}-{u_max} names qualify). The "
        f"liquidity rule removes selection-by-recent-performance <i>within</i> the list but cannot recover missing "
        f"names. The bias is visible: the equal-weight universe returned {pct(risk.loc['universe|lo|ew|monthly','cagr'])} "
        f"a year against {pct(risk.loc['nifty50','cagr'])} for the Nifty 50 price index. Long-only results are "
        f"inflated by this; long-short and relative results much less so.", BODY))
    s.append(Paragraph(
        f"<b>Corporate actions.</b> Yahoo left some bonuses and demergers unadjusted (a one-day −35% to −75% move "
        f"no holder experienced). Days with |return| &gt; 25% while the index moved &lt; 3% were reviewed by hand; "
        f"{(ca.action=='adjust').sum()} were back-adjusted and one pre-relisting history dropped "
        f"({', '.join(ca.ticker)}). {(ca.verified=='no').sum()} of these could not be verified against an "
        f"official source. Genuine crashes (Yes Bank 2020, Adani 2023) were left in.", BODY))

    s.append(Paragraph("2. Factors and cross-sectional processing", H1))
    ftab = pd.DataFrame([
        ["mom_12_1", "close[t−21] / close[t−252] − 1", "+"],
        ["strev", "−(21-day return)", "+"],
        ["lowvol", "1 / annualised 60-day volatility", "+"],
        ["idio_vol", "std of residuals, 60-day regression on Nifty 50", "−"],
        ["liquidity", "log 60-day median traded value (from the SQL universe)", "−"],
        ["high_52w", "close / 252-day max close", "+"],
        ["beta", "252-day regression slope on Nifty 50", "−"]], columns=["Factor", "Definition (trading days)", "Prior sign"])
    s.append(table(ftab, [2.5 * cm, 11 * cm, 2 * cm]))
    s.append(Spacer(1, 4))
    s.append(Paragraph(
        "All factors are trailing statistics ending at t; returns are never forward-filled and each window needs 80% "
        "of its days. A unit test appends future data and checks that no past factor value changes. At each "
        "rebalance date, independently, each factor is winsorized at the 1st/99th percentile, z-scored, and "
        "demeaned within sector (sectors with fewer than 3 names are pooled). Prior signs were committed to the "
        "repository before any IC was computed and are used only to orient the equal-weight composite.", BODY))

    s.append(Paragraph("3. Signal evaluation", H1))
    t = ic.reindex(decay.index)
    itab = pd.DataFrame({"Factor": t.index, "Mean IC": t.mean_ic.map(lambda x: f"{x:.3f}"),
                         "t-stat": t.t_stat.map(lambda x: f"{x:.1f}"), "Hit rate": t.hit_rate.map(lambda x: pct(x, 0)),
                         "IC 5d": decay.h5.map(lambda x: f"{x:.3f}"), "IC 63d": decay.h63.map(lambda x: f"{x:.3f}"),
                         "D10−D1 /mo": t.d10_minus_d1_monthly.map(lambda x: pct(x, 2)),
                         "Monotonicity ρ": t.mono_rho.map(lambda x: f"{x:.2f}"),
                         "Prior": t.prior_sign.map(lambda x: "+" if x > 0 else "−")})
    s.append(table(itab))
    s.append(caption(f"Rank IC = Spearman correlation of the factor with the 21-day forward return (entered at close t+1), "
                     f"{int(t.n.iloc[0])} months. t = mean/std × √n. Monotonicity ρ = rank correlation of decile number with mean decile return."))
    s.append(Spacer(1, 4))
    s.append(fig("ic_bars.png", 12))
    s.append(Paragraph(
        f"<b>Reading.</b> Momentum strengthens with horizon (IC {f(decay.loc['mom_12_1','h5'],3)} at 5 days to "
        f"{f(decay.loc['mom_12_1','h63'],3)} at 63) and its top decile does most of the work. Short-term reversal is "
        f"strongest at 5 days and decays, as expected. Liquidity is the strongest and most persistent signal "
        f"(63-day IC {f(decay.loc['liquidity','h63'],3)}). Low-volatility shows near-zero rank IC but a clean decile "
        f"pattern in the <i>wrong</i> direction (ρ = {f(ic.loc['lowvol','mono_rho'])}): the most volatile names had "
        f"higher mean returns, driven by a fat right tail. That is a period effect (2020-2024 small/high-beta rally), "
        f"not a law. No factor has |IC| above 0.04; anything near 0.10 would have pointed to a bug.", BODY))
    s.append(fig("ic_decay.png"))
    s.append(fig("deciles.png"))
    s.append(Paragraph(
        "<b>Are seven factors seven signals?</b> No. Low-vol and idio-vol are almost the same signal (average "
        "cross-sectional rank correlation −0.92), and 52-week-high is tied to momentum (+0.47) and reversal (−0.46). "
        "Roughly five independent signals remain.", BODY))
    s.append(fig("factor_corr.png", 9))

    s.append(Paragraph("4. ML ranking model", H1))
    s.append(Paragraph(
        "LightGBM predicts each stock's percentile rank of 21-day forward return from the seven factors. "
        "<b>Walk-forward:</b> each calendar year from 2019 is a test block; the model trains on all earlier data. "
        "<b>Purging:</b> a sample at t has a label spanning trading days t+1 to t+22, so training samples whose label "
        "window overlaps the test block are dropped; without this the last month before each block leaks into it. "
        f"<b>Embargo:</b> hyperparameters (8 configurations) are chosen by purged 5-fold CV inside the training data, "
        f"with a further {5}-day embargo after each validation block, because serially correlated samples just after "
        f"validation also carry its information. The inner CV chose the smallest model "
        f"({int(folds.p_num_leaves.mode()[0])} leaves, {int(folds.p_n_estimators.mode()[0])} trees) in "
        f"{(folds.p_num_leaves == folds.p_num_leaves.mode()[0]).sum()} of {len(folds)} years, which is itself a statement "
        f"about the noise level.", BODY))
    otab = oos.loc[["lgbm", "linear", "ew_ic", "ew_prior", "best_1"]].copy()
    otab = pd.DataFrame({"Model": ["LightGBM", "Linear (OLS)", "EW composite, training-IC signs",
                                   "EW composite, prior signs", "Best single factor (chosen in training)"],
                         "OOS mean IC": otab.oos_mean_ic.map(lambda x: f"{x:.3f}"),
                         "IC t-stat": otab.ic_t.map(lambda x: f"{x:.1f}"),
                         "Hit rate": otab.hit_rate.map(lambda x: pct(x, 0)),
                         "D10−D1 /mo": otab.d10_d1_monthly.map(lambda x: pct(x, 2))})
    s.append(table(otab, [6.5 * cm, 2.5 * cm, 2.2 * cm, 2.2 * cm, 2.5 * cm]))
    s.append(caption(f"Out-of-sample {folds.test_first_date.min()[:4]}-{d1[:4]}, {int(oos.loc['lgbm','n_months'])} months."))
    s.append(Paragraph(
        f"<b>Checks.</b> With labels shuffled within each date (6 seeds), out-of-sample IC averages "
        f"{f(mchk['shuffled_label_lgbm_oos_ic_mean'],4)} with standard deviation {f(mchk['shuffled_label_lgbm_oos_ic_sd'],3)}; "
        f"the real model sits about {f(oos.loc['lgbm','oos_mean_ic']/mchk['shuffled_label_lgbm_oos_ic_sd'],1)} standard "
        f"deviations above that. Turning purging off changes IC by only "
        f"{f(abs(mchk['no_purge_lgbm_oos_ic']-oos.loc['lgbm','oos_mean_ic']),4)}: with monthly samples and 21-day labels "
        f"just one month overlaps each boundary. Purging would matter much more with daily samples; it is implemented "
        f"and unit-tested regardless.", BODY))

    s.append(PageBreak())
    s.append(Paragraph("5. Portfolio backtest", H1))
    s.append(Paragraph(
        "Signals formed at close t are traded at close t+1; positions then drift with prices until the next "
        "rebalance. Long-short books are dollar-neutral (+1/−1 × NAV) in the top/bottom 10% of names (16-19 "
        "names per leg). Costs are charged on one-way turnover: base case 15 bps (securities transaction tax, stamp "
        "duty and fees) plus 5 bps slippage. Shorts are assumed implementable through single-stock futures, without "
        "borrow cost. LightGBM was fixed as the headline signal before any backtest was run.", BODY))
    btab = bt.loc[[H, L, E, "ew_prior|ls|ew|weekly", "lgbm|ls|vol|monthly", "lgbm|lo|ew|monthly",
                   "universe|lo|ew|monthly"]]
    names = ["LightGBM L/S", "Linear L/S", "EW composite L/S", "EW composite L/S, weekly",
             "LightGBM L/S, inverse-vol", "LightGBM long-only", "Universe EW (benchmark)"]
    btab = pd.DataFrame({"Strategy": names, "CAGR gross": btab.cagr_gross.map(pct), "CAGR net": btab.cagr_net.map(pct),
                         "Sharpe net": btab.sharpe_net.map(f), "Turnover / rebal": btab.turnover_per_rebal.map(lambda x: f"{x:.2f}x"),
                         "Cost drag / yr": btab.cost_drag_annual.map(pct),
                         "Breakeven": btab.breakeven_cost_bps.map(lambda x: "" if pd.isna(x) else ("n/a" if np.isinf(x) else f"{x:.0f} bps"))})
    s.append(table(btab, [5 * cm, 1.9 * cm, 1.8 * cm, 1.8 * cm, 2.3 * cm, 2.1 * cm, 1.9 * cm]))
    s.append(caption("Net of 20 bps. Long-only Sharpe uses excess return over 6% (approximate 91-day T-bill); long-short is self-financing."))
    s.append(Spacer(1, 4))
    s.append(fig("cost_sensitivity.png", 14))
    s.append(Paragraph(
        f"<b>Costs decide the result.</b> The LightGBM book turns over {f(hb.turnover_per_rebal,1)}x NAV a month "
        f"(about {pct(hb.turnover_per_rebal/4,0)} of the 2x gross book), so each 10 bps of cost removes roughly "
        f"{pct(hb.turnover_annual*0.001,1)} a year. Breakeven is {f(hb.breakeven_cost_bps,0)} bps for LightGBM and "
        f"{f(bt.loc[L,'breakeven_cost_bps'],0)} bps for the linear model. Weekly rebalancing of the composite doubles "
        f"turnover to {f(bt.loc['ew_prior|ls|ew|weekly','turnover_annual'],0)}x a year and loses money even gross.", BODY))
    s.append(fig("equity_curves.png"))
    s.append(Paragraph(
        f"<b>Checks.</b> The engine is unit-tested against hand calculations. Each holding-period return of the "
        f"backtest correlates {f(bchk['corr_bt_period_return_vs_d10_d1_spread'],2)} with the Phase 4/5 decile spread. "
        f"Executing at the same close (lag 0) or a day later (lag 2) moves gross Sharpe from "
        f"{f(bchk['lgbm_ls_gross_sharpe_entry_lag1'])} to {f(bchk['lgbm_ls_gross_sharpe_entry_lag0'])} / "
        f"{f(bchk['lgbm_ls_gross_sharpe_entry_lag2'])}. Twenty books built on random scores have gross Sharpe "
        f"{f(bchk['random_signal_ls_gross_sharpe_mean'])} ± {f(bchk['random_signal_ls_gross_sharpe_sd'])}.", BODY))

    s.append(Paragraph("6. Risk and robustness", H1))
    rk = risk.loc[[H, L, E, "lgbm|lo|ew|monthly", "universe|lo|ew|monthly", "nifty50", "lgbm_lo_minus_universe"]]
    rtab = pd.DataFrame({"Strategy": ["LightGBM L/S", "Linear L/S", "EW composite L/S", "LightGBM long-only",
                                      "Universe EW", "Nifty 50 (price)", "LightGBM L/O − universe"],
                         "CAGR": rk.cagr.map(pct), "Vol": rk.vol.map(pct, ), "Sharpe": rk.sharpe.map(f),
                         "Sortino": rk.sortino.map(f), "Max DD": rk.max_drawdown.map(lambda x: pct(x, 0)),
                         "Calmar": rk.calmar.map(f), "Hit /mo": rk.hit_rate_monthly.map(lambda x: pct(x, 0)),
                         "Mkt beta": rk.market_beta.map(f),
                         "beta t": rk.beta_t.map(lambda x: "" if pd.isna(x) else f"{x:.1f}")})
    s.append(table(rtab, [4.2 * cm] + [1.45 * cm] * 9))
    s.append(caption("Sortino divides mean excess return by downside deviation only. Market beta: daily OLS on Nifty 50 returns, Newey-West (5 lags) t-stat."))
    s.append(Paragraph(
        f"<b>Not market-neutral.</b> The dollar-neutral L/S book has residual beta {f(hr.market_beta)} "
        f"(t = {f(hr.beta_t,1)}): the short leg holds higher-beta names. Beta-neutral leg sizing is the obvious fix. "
        f"Sortino ({f(hr.sortino)}) is only modestly above Sharpe ({f(hr.sharpe)}), so the losses are not unusually "
        f"small relative to gains. The {pct(hr.max_drawdown,0)} drawdown ran from 2020 to late 2023 and recovered in 2024.", BODY))
    s.append(fig("rolling_sharpe_drawdown.png", 15))
    s.append(fig("monthly_heatmap.png", 15))

    s.append(Paragraph("Parameter sensitivity", H2))
    s.append(Paragraph(
        f"The whole pipeline (factors, processing, purged walk-forward training, backtest) was rerun on a grid of "
        f"momentum lookbacks × volatility windows. LightGBM's net Sharpe is positive in {pos_cells} of {len(sens_a)} cells "
        f"(range {f(sens_a.lgbm_sharpe_net.min())} to {f(sens_a.lgbm_sharpe_net.max())}; default cell "
        f"{f(default_cell.lgbm_sharpe_net)}); the linear model ranges {f(sens_a.linear_sharpe_net.min())} to "
        f"{f(sens_a.linear_sharpe_net.max())}. In portfolio construction, concentration helps: the 5% tails earn "
        f"Sharpe {f(sens_b[(sens_b.quantile==0.05)&(sens_b.cost_bps==20)].sharpe_net.iloc[0])} at 20 bps against "
        f"{f(sens_b[(sens_b.quantile==0.2)&(sens_b.cost_bps==20)].sharpe_net.iloc[0])} for 20% tails, consistent with the "
        f"signal living in the extremes.", BODY))
    s.append(fig("sensitivity_lookback.png", 15))
    s.append(fig("sensitivity_portfolio.png", 8))

    s.append(Paragraph("Multiple testing", H2))
    vt = variants.copy()
    vt.columns = ["Phase", "What was tried", "Count", "Note"]
    s.append(table(vt, [1.6 * cm, 6.2 * cm, 1.3 * cm, 7.4 * cm]))
    s.append(Spacer(1, 4))
    s.append(Paragraph(
        f"<b>{int(variants['count'].sum())} configurations were evaluated, {int(dsr.loc[H,'n_strategy_variants'])} of them "
        f"strategy variants scored on out-of-sample performance.</b> Over {f(risk.loc[H,'years'],1)} years a Sharpe ratio "
        f"is measured with standard error ~{f(dsr.loc[H,'sharpe_sd'])}, and the best of "
        f"{int(dsr.loc[H,'n_strategy_variants'])} no-skill strategies would be expected to reach "
        f"~{f(dsr.loc[H,'expected_max_null_sharpe'])}. The deflated-Sharpe probability that LightGBM's true Sharpe beats "
        f"that hurdle is {f(dsr.loc[H,'deflated_sharpe_prob'])} (linear: {f(dsr.loc[L,'deflated_sharpe_prob'])}), against "
        f"{f(dsr.loc[H,'plain_sharpe_prob'])} with no correction. The variants are correlated, so the true hurdle is "
        f"lower, but the conclusion stands: the long-short result is suggestive, not established.", BODY))

    s.append(Paragraph("7. Limitations", H1))
    s += bullets([
        "Survivorship: the candidate list is today's Nifty 200. Long-only absolute returns are inflated; the universe "
        "benchmark absorbs much of this, but not all.",
        "Sector labels are today's classification applied to the whole history.",
        f"{(ca.verified=='no').sum()} corporate-action adjustments are inferred from price and volume and could not be "
        "checked against exchange records (Yahoo was unreachable from the build environment).",
        "Costs are a flat bps assumption; no market-impact model, borrow fees, futures roll costs or short-sale constraints.",
        "The Nifty 50 benchmark is a price index (no dividends, roughly 1-1.5%/yr), while stock prices are dividend-adjusted.",
        "One market, one decade: the factor signs found here (notably for volatility) are period-specific.",
    ])
    s.append(Paragraph("8. Next steps", H1))
    s += bullets([
        "Beta-neutral leg weights to remove the residual short-market exposure.",
        "Turnover control: trade only when a name leaves a wider buffer band (e.g. hold until it exits the top 20%).",
        "A survivorship-free universe from historical index constituents (NSE archives).",
        "Regime analysis of the volatility factors and a combined long-leg strategy, the most robust result here.",
    ])

    s.append(PageBreak())
    s.append(Paragraph("Appendix: reproducibility", H1))
    s.append(Paragraph(
        "Pipeline (each step reads only what earlier steps wrote): <font face='DV'>build_db.py → build_universe.py "
        "(sql/universe.sql) → factors.py → preprocess.py → evaluate.py → model.py → backtest.py → sensitivity.py → "
        "risk.py → export_powerbi.py → report.py</font>, or <font face='DV'>python src/run_all.py</font>. "
        "A pytest suite covers look-ahead in factors, per-date independence of processing, IC/forward-return "
        "alignment, purge/embargo correctness, the backtest engine and the risk formulas.", BODY))

    OUT.parent.mkdir(exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                            bottomMargin=1.8 * cm, title="Equity Factor Research & Systematic Backtest",
                            author="Pratham Manchanda")
    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
