import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta
import plotly.graph_objects as go

st.set_page_config("Trading Dashboard Pro", layout="wide")

# ------------------ UI ------------------

st.title("📈 Pro Trading Dashboard")

TIMEFRAMES = {
    "24 Hours": ("1d", "1h"),
    "1 Week": ("5d", "1h"),
    "1 Month": ("1mo", "1d"),
    "6 Months": ("6mo", "1d"),
    "1 Year": ("1y", "1d"),
    "3 Years": ("3y", "1d"),
    "5 Years": ("5y", "1d"),
}

ticker = st.sidebar.text_input("Ticker", "AAPL").upper()
range_label = st.sidebar.selectbox("Time Range", TIMEFRAMES.keys())
forecast_days = st.sidebar.slider("Forecast Days", 1, 60, 5)
chart_type = st.sidebar.radio("Chart Type", ["Candles", "Line"])

# ------------------ DATA ------------------

def load_data(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    df.dropna(inplace=True)
    return df

period, interval = TIMEFRAMES[range_label]
df = load_data(ticker, period, interval)

if df.empty:
    st.stop()

# ------------------ INDICATORS (CRASH PROOF) ------------------

def add_indicators(df):
    df = df.copy()

    df["MA20"] = df["Close"].rolling(20, min_periods=1).mean()
    df["MA50"] = df["Close"].rolling(50, min_periods=1).mean()

    std = df["Close"].rolling(20, min_periods=1).std()

    # IMPORTANT: std is Series aligned to df — NO shape issues
    df["BB_upper"] = df["MA20"] + 2 * std
    df["BB_lower"] = df["MA20"] - 2 * std

    delta = df["Close"].diff().fillna(0)
    gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss = -delta.clip(upper=0).rolling(14, min_periods=1).mean()

    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()

    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9).mean()

    return df

df = add_indicators(df)

# ------------------ FORECAST ------------------

def forecast(df, days):
    x = np.arange(len(df))
    y = df["Close"].values

    coef = np.polyfit(x, y, 1)
    future_x = np.arange(len(df), len(df)+days)

    preds = coef[0]*future_x + coef[1]
    dates = [df.index[-1] + timedelta(days=i+1) for i in range(days)]

    return dates, preds

forecast_dates, forecast_values = forecast(df, forecast_days)

# ------------------ PAPER TRADING ------------------

if "balance" not in st.session_state:
    st.session_state.balance = 10000
    st.session_state.shares = 0
    st.session_state.trades = []

price = float(df["Close"].iloc[-1])

# ------------------ CHART ------------------

def plot_chart():

    ymin = min(df["Low"].min(), df["BB_lower"].min(), np.min(forecast_values)) - 1
    ymax = max(df["High"].max(), df["BB_upper"].max(), np.max(forecast_values)) + 1

    fig = go.Figure()

    if chart_type == "Candles":
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price"
        ))
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"))

    fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], name="MA50"))

    fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], line=dict(dash="dot"), name="BB Upper"))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], line=dict(dash="dot"), name="BB Lower"))

    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_values,
        line=dict(dash="dash", color="yellow"),
        name="Forecast"
    ))

    for t in st.session_state.trades:
        color = "green" if t["type"]=="BUY" else "red"
        fig.add_trace(go.Scatter(
            x=[t["time"]],
            y=[t["price"]],
            mode="markers",
            marker=dict(color=color, size=12),
            name=t["type"]
        ))

    fig.update_layout(
        height=650,
        template="plotly_dark",
        yaxis=dict(range=[ymin, ymax]),
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)

# ------------------ TABS ------------------

market, sim = st.tabs(["📊 Market", "💰 Trading Simulator"])

with market:
    plot_chart()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f"${price:.2f}")
    c2.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")
    c3.metric("MACD", f"{df['MACD'].iloc[-1]:.2f}")
    c4.metric("Signal", f"{df['Signal'].iloc[-1]:.2f}")

    st.subheader("RSI")
    st.line_chart(df["RSI"])

    st.subheader("MACD")
    st.line_chart(df[["MACD","Signal"]])

# ------------------ SIMULATOR ------------------

with sim:
    st.subheader("Paper Trading")

    col1, col2, col3 = st.columns(3)
    col1.metric("Balance", f"${st.session_state.balance:.2f}")
    col2.metric("Shares", st.session_state.shares)
    col3.metric("Price", f"${price:.2f}")

    qty = st.number_input("Shares", 1, 1000, 1)

    buy = st.button("BUY")
    sell = st.button("SELL")

    if buy and st.session_state.balance >= qty * price:
        st.session_state.balance -= qty * price
        st.session_state.shares += qty
        st.session_state.trades.append({
            "type":"BUY",
            "price":price,
            "time":df.index[-1]
        })

    if sell and st.session_state.shares >= qty:
        st.session_state.balance += qty * price
        st.session_state.shares -= qty
        st.session_state.trades.append({
            "type":"SELL",
            "price":price,
            "time":df.index[-1]
        })

    pnl = st.session_state.balance + st.session_state.shares*price - 10000
    st.metric("Profit / Loss", f"${pnl:.2f}")

    if st.session_state.trades:
        st.table(pd.DataFrame(st.session_state.trades))
