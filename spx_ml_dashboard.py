import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="AI Trading Dashboard Pro")

# ---------------------------
# CONFIG
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

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def load_data(ticker, period):
    interval = "1h" if period in ["1d", "5d"] else "1d"
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    df.dropna(inplace=True)
    if df.empty:
        st.warning(f"No data available for {ticker} with period {period}.")
    return df

def add_indicators(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    # Moving averages
    df["MA20"] = df["Close"].rolling(20, min_periods=1).mean()
    df["MA50"] = df["Close"].rolling(50, min_periods=1).mean()

    # Bollinger Bands
    rolling_std = df["Close"].rolling(20, min_periods=1).std().reindex(df.index).fillna(0)
    df["BB_upper"] = df["MA20"] + 2 * rolling_std
    df["BB_lower"] = df["MA20"] - 2 * rolling_std

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss = -delta.clip(upper=0).rolling(14, min_periods=1).mean()
    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    df.fillna(method="bfill", inplace=True)
    df.fillna(method="ffill", inplace=True)
    return df

def forecast(df, days_ahead):
    if len(df) < 2:
        return [], np.array([])
    trend = np.polyfit(range(len(df)), df["Close"], 1)
    future_x = np.arange(len(df), len(df) + days_ahead)
    preds = trend[0] * future_x + trend[1]
    future_dates = [df.index[-1] + timedelta(days=i+1) for i in range(days_ahead)]
    return future_dates, preds

def plot_chart(df, forecast_dates=None, forecast_values=None, trades=None, chart_type="Line"):
    ymin = df["Low"].min()
    ymax = df["High"].max()

    if forecast_values is not None and len(forecast_values) > 0:
        ymin = min(ymin, np.min(forecast_values))
        ymax = max(ymax, np.max(forecast_values))

    if "BB_lower" in df.columns and "BB_upper" in df.columns:
        ymin = min(ymin, df["BB_lower"].min())
        ymax = max(ymax, df["BB_upper"].max())

    pad = (ymax - ymin) * 0.05
    ymin -= pad
    ymax += pad

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

    # Indicators
    fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], mode="lines", name="MA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], mode="lines", name="MA50"))

    if "BB_upper" in df.columns and "BB_lower" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], mode="lines", line=dict(dash="dot", color="cyan"), name="BB Upper"))
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], mode="lines", line=dict(dash="dot", color="cyan"), name="BB Lower"))

    if forecast_values is not None and forecast_dates is not None and len(forecast_values) > 0:
        fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_values, mode="lines", line=dict(dash="dash", color="yellow"), name="Forecast"))

    # Paper trades
    if trades:
        for t_type, t_price, t_qty, t_time in trades:
            color = "green" if t_type=="BUY" else "red"
            fig.add_trace(go.Scatter(
                x=[t_time],
                y=[t_price],
                mode="markers+text",
                marker=dict(color=color, size=12, symbol="triangle-up" if t_type=="BUY" else "triangle-down"),
                text=[f"{t_type} {t_qty}"],
                textposition="top center",
                name=f"{t_type} Trade"
            ))

    fig.update_layout(height=600, yaxis=dict(range=[ymin, ymax]), xaxis_rangeslider_visible=False, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.title("AI Trading Dashboard Pro")
ticker = st.sidebar.text_input("Ticker", "AAPL").upper()
historical_range_label = st.sidebar.selectbox("Historical Data Range", list(TIMEFRAMES.keys()))
days_ahead = st.sidebar.number_input("Forecast Days Ahead", 1, 90, value=5)
chart_type = st.sidebar.radio("Chart Type", ["Candles", "Line"])
show_rsi = st.sidebar.checkbox("Show RSI", True)
show_macd = st.sidebar.checkbox("Show MACD", True)
show_bollinger = st.sidebar.checkbox("Show Bollinger Bands", True)
show_forecast = st.sidebar.checkbox("Show Forecast", True)

# ---------------------------
# TABS
# ---------------------------
tabs = st.tabs(["📊 Market", "💰 Paper Trading Simulator"])

# ---------------------------
# LOAD DATA & INDICATORS
# ---------------------------
df = load_data(ticker, TIMEFRAMES[historical_range_label])
if df.empty:
    st.stop()

df = add_indicators(df)
forecast_dates, forecast_values = forecast(df, days_ahead)

# ---------------------------
# MARKET TAB
# ---------------------------
with tabs[0]:
    st.subheader(f"{ticker} Market Data")
    plot_chart(df, forecast_dates if show_forecast else None, forecast_values if show_forecast else None, trades=st.session_state.get("trades", []), chart_type=chart_type)

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
# PAPER TRADING SIMULATOR
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
        st.session_state.trades.append(("BUY", price, qty, df.index[-1]))

    if sell and st.session_state.shares >= qty:
        st.session_state.balance += price * qty
        st.session_state.shares -= qty
        st.session_state.trades.append(("SELL", price, qty, df.index[-1]))

    pnl = st.session_state.balance + st.session_state.shares * price - 10000
    st.metric("Profit / Loss", f"${pnl:.2f}")

    if st.session_state.trades:
        st.subheader("Trade Log")
        st.table(pd.DataFrame(st.session_state.trades, columns=["Type","Price","Shares","Time"]))


