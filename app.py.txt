import streamlit as st
import plotly.graph_objects as go

from data import load
from features import build
from models import train
from strategies import apply
from risk import position_size
from backtest import run

st.set_page_config(layout="wide")

st.sidebar.title("🏦 Institutional AI Trading System")

ticker = st.sidebar.text_input("Ticker", "AAPL")
period = st.sidebar.selectbox("Range", ["1y","3y","5y"])

df = load(ticker, period)
df = build(df)
df = train(df)
df = apply(df)
df = position_size(df)
df, stats = run(df)

tabs = st.tabs([
    "📊 Trading Signals",
    "📈 Portfolio Backtest",
    "📋 Strategy Analytics"
])

# ===== CHART ===== #

with tabs[0]:
    fig = go.Figure()

    fig.add_candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"]
    )

    buys = df[df["Signal"] == 1]

    fig.add_scatter(
        x=buys.index,
        y=buys["Close"],
        mode="markers",
        marker=dict(color="lime", size=6),
        name="AI Entry"
    )

    fig.update_layout(template="plotly_dark", height=600)
    st.plotly_chart(fig, use_container_width=True)

# ===== BACKTEST ===== #

with tabs[1]:
    st.subheader("Equity Curve")
    st.line_chart(df["Equity"])

# ===== ANALYTICS ===== #

with tabs[2]:
    for k, v in stats.items():
        st.metric(k, f"{v:.2f}")
