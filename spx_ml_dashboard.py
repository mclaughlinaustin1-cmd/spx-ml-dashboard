import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(layout="wide", page_title="AI Trading Dashboard")

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------

TIMEFRAMES = {
    "24 Hours": "1d",
    "1 Week": "5d",
    "1 Month": "1mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "3 Years": "3y",
    "5 Years": "5y"
}

def load_data(ticker, period):
    interval = "1h" if period in ["1d", "5d"] else "1d"
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    df.dropna(inplace=True)
    return df

def add_indicators(df):
    df = df.copy()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100/(1+rs))

    # MACD
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9).mean()

    return df

def forecast(df, days_ahead):
    # Simple linear trend forecast
    trend = np.polyfit(range(len(df)), df["Close"], 1)
    future_x = np.arange(len(df), len(df)+days_ahead)
    preds = trend[0]*future_x + trend[1]
    future_dates = [df.index[-1] + timedelta(days=i+1) for i in range(days_ahead)]
    return future_dates, preds

# ---------------------------
# SIDEBAR UI
# ---------------------------

st.sidebar.title("AI Trading Dashboard")
ticker = st.sidebar.text_input("Ticker", "AAPL").upper()
historical_range_label = st.sidebar.selectbox("Historical Data Range", list(TIMEFRAMES.keys()))
days_ahead = st.sidebar.number_input("Predict Days Ahead", min_value=1, max_value=90, value=5, step=1)
chart_type = st.sidebar.radio("Chart Type", ["Candles", "Line"])
show_rsi = st.sidebar.checkbox("Show RSI", True)
show_macd = st.sidebar.checkbox("Show MACD", True)
show_forecast = st.sidebar.checkbox("Show Forecast", True)

# ---------------------------
# TABS
# ---------------------------
tabs = st.tabs(["📊 Market", "💰 Paper Trading Simulator"])

# ---------------------------
# LOAD DATA
# ---------------------------
df = load_data(ticker, TIMEFRAMES[historical_range_label])
df = add_indicators(df)
future_dates, forecast_values = forecast(df, days_ahead)

# ---------------------------
# AUTOSCALE Y-AXIS
# ---------------------------
ymin = min(df["Low"].min(), forecast_values.min()) - 1
ymax = max(df["High"].max(), forecast_values.max()) + 1

# ---------------------------
# MARKET TAB
# ---------------------------
with tabs[0]:
    st.subheader(f"{ticker} Market Data")

    fig = go.Figure()
    if chart_type == "Candles":
        fig.add_candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price"
        )
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"))

    fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], mode="lines", name="MA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], mode="lines", name="MA50"))

    if show_forecast:
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=forecast_values,
            mode="lines",
            name="Forecast",
            line=dict(dash="dash", color="yellow")
        ))

    fig.update_layout(
        height=600,
        yaxis=dict(range=[ymin, ymax]),
        xaxis_rangeslider_visible=False,
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Price", f"${df['Close'].iloc[-1]:.2f}")
    col2.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")
    col3.metric("MACD", f"{df['MACD'].iloc[-1]:.2f}")
    col4.metric("Signal", f"{df['Signal'].iloc[-1]:.2f}")

    if show_rsi:
        st.subheader("RSI")
        st.line_chart(df["RSI"])

    if show_macd:
        st.subheader("MACD")
        st.line_chart(df[["MACD","Signal"]])

# ---------------------------
# PAPER TRADING SIMULATOR TAB
# ---------------------------
if "balance" not in st.session_state:
    st.session_state.balance = 10000
    st.session_state.shares = 0
    st.session_state.trades = []

with tabs[1]:
    st.subheader("💰 Paper Trading Simulator")
    price = df["Close"].iloc[-1]

    col1, col2, col3 = st.columns(3)
    col1.metric("Balance", f"${st.session_state.balance:.2f}")
    col2.metric("Shares", st.session_state.shares)
    col3.metric("Current Price", f"${price:.2f}")

    qty = st.number_input("Shares to Trade", 1, 1000, 1)

    buy = st.button("📈 Buy")
    sell = st.button("📉 Sell")

    if buy and st.session_state.balance >= price * qty:
        st.session_state.balance -= price * qty
        st.session_state.shares += qty
        st.session_state.trades.append(("BUY", price, qty, datetime.now()))

    if sell and st.session_state.shares >= qty:
        st.session_state.balance += price * qty
        st.session_state.shares -= qty
        st.session_state.trades.append(("SELL", price, qty, datetime.now()))

    pnl = st.session_state.balance + st.session_state.shares * price - 10000
    st.metric("Profit / Loss", f"${pnl:.2f}")

    if st.session_state.trades:
        st.subheader("Trade Log")
        st.table(pd.DataFrame(
            st.session_state.trades,
            columns=["Type","Price","Shares","Time"]
        ))

