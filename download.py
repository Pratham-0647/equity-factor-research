import pandas as pd, yfinance as yf

df = pd.read_csv("nifty200.csv")
syms = df["Symbol"] + ".NS"

px = yf.download(list(syms) + ["^NSEI"], start="2015-01-01",
                 auto_adjust=True, group_by="ticker")
px.to_parquet("prices.parquet")

df[["Symbol", "Industry"]].to_csv("sectors.csv", index=False)
