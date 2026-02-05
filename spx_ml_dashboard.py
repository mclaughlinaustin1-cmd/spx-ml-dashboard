import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta

st.set_page_config(page_title="Trading Dashboard", layout="wide")
st.title("📊 Stock Prediction Dashboard")

# ========== SIDEBAR ==========

ticker = st.sidebar.text_input("Ticker Symbol", "SPY")
history_days = st.sidebar.slider("Historical Days", 30, 2000, 365)
forecast_days = st.sidebar.slider("Forecast Days", 1, 90, 14)
chart_mode = st.sidebar.radio("Chart Type", ["Candlestick", "Line"])

# ========== DATA ==========

@st.cache_data
def fetch_data(symbol, days):
    df = yf.download(symbol, period=f"{days}d", progress=False)
    df.reset_index(inplace=True)
    return df

df = fetch_data(ticker, history_days)

if df.empty:
    st.error("No data loaded")
    st.stop()

close = df["Close"]

# ========== BOLLINGER BANDS (NO DF ASSIGNMENTS) ==========

ma20 = close.rolling(20, min_periods=1).mean()
std = close.rolling(20, min_periods=1).std()

bb_upper = ma20 + 2 * std
bb_lower = ma20 - 2 * std

# ========== SIMPLE FORECAST ==========

x = np.arange(len(close))
slope, intercept = np.polyfit(x, close.values, 1)

future_x = np.arange(len(close), len(close) + forecast_days)
forecast = slope * future_x + intercept

future_dates = [
    df["Date"].iloc[-1] + timedelta(days=i+1)
    for i in range(forecast_days)
]

# ========== CHART ==========

fig = go.Figure()

if chart_mode == "Candlestick":
    fig.add_trace(go.Candlestick(
        x=df["Date"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price"
    ))
else:
    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=close,
        name="Close",
        mode="lines"
    ))

fig.add_trace(go.Scatter(x=df["Date"], y=ma20, name="MA20"))
fig.add_trace(go.Scatter(x=df["Date"], y=bb_upper, name="BB Upper", line=dict(dash="dot")))
fig.add_trace(go.Scatter(x=df["Date"], y=bb_lower, name="BB Lower", line=dict(dash="dot")))

fig.add_trace(go.Scatter(
    x=future_dates,
    y=forecast,
    name="Forecast",
    line=dict(dash="dash", color="yellow")
))

fig.update_layout(
    template="plotly_dark",
    height=650,
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig, use_container_width=True)

# ========== METRICS ==========

col1, col2, col3 = st.columns(3)

col1.metric("Current Price", f"${close.iloc[-1]:.2f}")
col2.metric("MA20", f"${ma20.iloc[-1]:.2f}")
col3.metric("Forecast End", f"${forecast[-1]:.2f}")


