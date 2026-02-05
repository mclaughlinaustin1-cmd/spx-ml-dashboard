import numpy as np

def position_size(df, capital=100000, risk_per_trade=0.01):
    risk = capital * risk_per_trade
    vol = df["Vol"]

    size = risk / (vol * df["Close"])
    df["Size"] = size.clip(0, 1)

    return df

