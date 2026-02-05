import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="AI Trading Platform Pro")

# ---------------------------
# Helpers
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
    # Fix: yfinance returns Multi-Index columns; we flatten them below
    df = yf.download(ticker, period=period, interval="1h" if period in ["1d","5d"] else "1d")
    if df.empty:
        return None
    
    # Flatten columns (e.g., ('Close', 'AAPL') becomes 'Close')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df.dropna(inplace=True)
    return df

def indicators(df):
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100/(1+rs))

    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    return df

def naive_forecast(df, steps=20):
    # Ensure values are 1D arrays for polyfit
    y = df["Close"].values.flatten()
    x = np.arange(len(y))
    trend = np.polyfit(x, y, 1)
    
    future_x = np.arange(len(y), len(y) + steps)
    preds = trend[0] * future_x + trend[1]
    
    # Generate future timestamps
    last_date = df.index[-1]
    future_dates = [last_date + timedelta(days=i+1) for i in range(steps)]
    return future_dates, preds

# ---------------------------
# UI
# ---------------------------

st.title("📈 Institutional AI Trading Dashboard")

ticker = st.sidebar.text_input("Ticker", "AAPL").upper()
tf_label = st.sidebar.selectbox("Timeframe", list(TIMEFRAMES.keys()))
chart_type = st.sidebar.radio("Chart Type", ["Candles", "Line"])

show_rsi = st.sidebar.checkbox("Show RSI", True)
show_macd = st.sidebar.checkbox("Show MACD", True)
show_forecast = st.sidebar.checkbox("Show Forecast", True)

tabs = st.tabs(["📊 Market", "💰 Paper Trading Simulator"])

# ---------------------------
# Load Data & Process
# ---------------------------

df_raw = load_data(ticker, TIMEFRAMES[tf_label])

if df_raw is not None:
    df = indicators(df_raw.copy())
    future_dates, forecast = naive_forecast(df)

    # ---------------------------
    # FIX: Robust Autoscale Logic
    # ---------------------------
    ymin = float(df["Low"].min())
    ymax = float(df["High"].max())

    if show_forecast:
        # Convert to float to avoid truth-value ambiguity with Series
        f_min = float(np.min(forecast))
        f_max = float(np.max(forecast))
        ymin = min(ymin, f_min)
        ymax = max(ymax, f_max)

    pad = (ymax - ymin) * 0.05
    ymin -= pad
    ymax += pad

    # ---------------------------
    # Market Tab
    # ---------------------------
    with tabs[0]:
        fig = go.Figure()

        if chart_type == "Candles":
            fig.add_candlestick(
                x=df.index, open=df["Open"], high=df["High"],
                low=df["Low"], close=df["Close"], name="Price"
            )
        else:
            fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close"))

        fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20", line=dict(width=1.5)))
        fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], name="MA50", line=dict(width=1.5)))

        if show_forecast:
            fig.add_trace(go.Scatter(
                x=future_dates, y=forecast, name="Forecast",
                line=dict(dash="dash", color="cyan")
            ))

        fig.update_layout(
            height=600,
            yaxis=dict(range=[ymin, ymax], title="Price ($)"),
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            margin=dict(l=0, r=0, t=30, b=0)
        )

        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Price", f"${df['Close'].iloc[-1]:.2f}")
        col2.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")
        col3.metric("MACD", f"{df['MACD'].iloc[-1]:.2f}")
        col4.metric("Signal", f"{df['Signal'].iloc[-1]:.2f}")

        if show_rsi:
            st.subheader("Relative Strength Index (RSI)")
            st.line_chart(df["RSI"])

        if show_macd:
            st.subheader("MACD & Signal")
            st.line_chart(df[["MACD","Signal"]])

    # ---------------------------
    # Simulator Tab
    # ---------------------------
    if "balance" not in st.session_state:
        st.session_state.balance = 10000.0
        st.session_state.shares = 0
        st.session_state.trades = []

    with tabs[1]:
        st.subheader("💰 Trading Simulator")
        price = float(df["Close"].iloc[-1])

        c1, c2, c3 = st.columns(3)
        c1.metric("Balance", f"${st.session_state.balance:.2f}")
        c2.metric("Shares Owned", st.session_state.shares)
        c3.metric("Market Price", f"${price:.2f}")

        qty = st.number_input("Transaction Quantity", 1, 10000, 1)
        
        btn_col1, btn_col2 = st.columns(2)
        if btn_col1.button("📈 Buy", use_container_width=True):
            if st.session_state.balance >= price * qty:
                st.session_state.balance -= price * qty
                st.session_state.shares += qty
                st.session_state.trades.append({"Type": "BUY", "Price": price, "Shares": qty})
                st.rerun()
            else:
                st.error("Insufficient Balance")

        if btn_col2.button("📉 Sell", use_container_width=True):
            if st.session_state.shares >= qty:
                st.session_state.balance += price * qty
                st.session_state.shares -= qty
                st.session_state.trades.append({"Type": "SELL", "Price": price, "Shares": qty})
                st.rerun()
            else:
                st.error("Not enough shares")

        pnl = (st.session_state.balance + st.session_state.shares * price) - 10000
        st.metric("Total Profit/Loss", f"${pnl:.2f}", delta=f"{pnl:.2f}")

        if st.session_state.trades:
            st.subheader("Trade Log")
            st.table(pd.DataFrame(st.session_state.trades))
else:
    st.error("No data found for this ticker. Please check the symbol and try again.")



