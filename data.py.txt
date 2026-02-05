import yfinance as yf
import pandas as pd

def load(ticker, period):
    df = yf.download(ticker, period=period, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    return df.dropna()
