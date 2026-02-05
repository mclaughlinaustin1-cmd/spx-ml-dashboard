import numpy as np

def build(df):
    df = df.copy()

    df["Return"] = df["Close"].pct_change()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["Momentum"] = df["Close"] - df["Close"].shift(10)
    df["Vol"] = df["Return"].rolling(20).std()

    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    return df.dropna()
