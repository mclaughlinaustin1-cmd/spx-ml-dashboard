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
    df = yf.download(ticker, period=period, interval="1h" if period in ["1d","5d"] else "1d")
    if df.empty:
        return None
    
    # Flatten Multi-Index columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df.dropna(inplace=True)
    return df

def indicators(df):
    # Moving Averages
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    # Bollinger Bands
    std = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["MA20"] + (std * 2)
    df["BB_Lower"] = df["MA20"] - (std * 2)

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

def naive_forecast(df, steps=20):
    y = df["Close"].values.flatten()
    x = np.arange(len(y))
    trend = np.polyfit(x, y, 1)
    
    future_x = np.arange(len(y), len(y) + steps)
    preds = trend[0] * future_x + trend[1]
    
    last_date = df.index[-1]
    future_dates = [last_date + timedelta(days=i+1) for i in range(steps)]
    return future_dates, preds

# ---------------------------
# UI Sidebar
# ---------------------------

st.title("📈 Institutional AI Trading Dashboard")

ticker = st.sidebar.text_input("Ticker Symbol", "AAPL").upper()
tf_label = st.sidebar.selectbox("Timeframe", list(TIMEFRAMES.keys()))
chart_type = st.sidebar.radio("Main Chart Style", ["Candles", "Line"])

st.sidebar.markdown("---")
st.sidebar.subheader("Overlays")
show_bb = st.sidebar.checkbox("Bollinger Bands", True)
show_forecast = st.sidebar.checkbox("AI Trend Forecast", True)

st.sidebar.subheader("Sub-charts")
show_rsi = st.sidebar.checkbox("RSI", True)
show_macd = st.sidebar.checkbox("MACD", True)

tabs = st.tabs(["📊 Market Data", "💰 Paper Trading"])

# ---------------------------
# Logic Execution
# ---------------------------

df_raw = load_data(ticker, TIMEFRAMES[tf_label])

if df_raw is not None and len(df_raw) > 20:
    df = indicators(df_raw.copy())
    future_dates, forecast = naive_forecast(df)

    # Autoscale calculation
    ymin = float(df["Low"].min())
    ymax = float(df["High"].max())

    if show_bb:
        ymin = min(ymin, float(df["BB_Lower"].min()))
        ymax = max(ymax, float(df["BB_Upper"].max()))

    if show_forecast:
        ymin = min(ymin, float(np.min(forecast)))
        ymax = max(ymax, float(np.max(forecast)))

    pad = (ymax - ymin) * 0.05
    ymin -= pad
    ymax += pad

    # ---------------------------
    # Market Tab
    # ---------------------------
    with tabs[0]:
        fig = go.Figure()

        # Main Price Chart
        if chart_type == "Candles":
            fig.add_candlestick(
                x=df.index, open=df["Open"], high=df["High"],
                low=df["Low"], close=df["Close"], name="Price"
            )
        else:
            fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close Price", line=dict(color='white')))

        # Bollinger Bands Traces
        if show_bb:
            fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="BB Upper", line=dict(color='rgba(173, 216, 230, 0.4)', width=1)))
            fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="BB Lower", line=dict(color='rgba(173, 216, 230, 0.4)', width=1), fill='tonexty', fillcolor='rgba(173, 216, 230, 0.05)'))
            fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="BB Middle (MA20)", line=dict(color='rgba(255, 255, 255, 0.3)', dash='dot')))

        # MA 50 Overlay
        fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], name="MA50", line=dict(color='orange', width=1.5)))

        if show_forecast:
            fig.add_trace(go.Scatter(
                x=future_dates, y=forecast, name="Linear Forecast",
                line=dict(dash="dash", color="cyan")
            ))

        fig.update_layout(
            height=650,
            yaxis=dict(range=[ymin, ymax], title="USD ($)"),
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            margin=dict(l=0, r=0, t=30, b=0)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
        m2.metric("RSI (14)", f"{df['RSI'].iloc[-1]:.1f}")
        m3.metric("MACD", f"{df['MACD'].iloc[-1]:.2f}")
        m4.metric("BB Width", f"{(df['BB_Upper'].iloc[-1] - df['BB_Lower'].iloc[-1]):.2f}")

        if show_rsi:
            st.subheader("RSI (14)")
            st.line_chart(df["RSI"])

        if show_macd:
            st.subheader("MACD")
            st.line_chart(df[["MACD","Signal"]])

    # ---------------------------
    # Simulator Tab
    # ---------------------------
    if "balance" not in st.session_state:
        st.session_state.balance = 10000.0
        st.session_state.shares = 0
        st.session_state.trades = []

    with tabs[1]:
        st.subheader("💰 Paper Trading Engine")
        price = float(df["Close"].iloc[-1])

        c1, c2, c3 = st.columns(3)
        c1.metric("Available Cash", f"${st.session_state.balance:.2f}")
        c2.metric("Position Size", f"{st.session_state.shares} units")
        c3.metric("Market Price", f"${price:.2f}")

        qty = st.number_input("Units to Trade", 1, 10000, 1)
        
        b1, b2 = st.columns(2)
        if b1.button("Buy Long", use_container_width=True):
            if st.session_state.balance >= price * qty:
                st.session_state.balance -= price * qty
                st.session_state.shares += qty
                st.session_state.trades.append({"Type": "BUY", "Price": price, "Units": qty, "Time": datetime.now().strftime("%H:%M:%S")})
                st.rerun()

        if b2.button("Sell/Close", use_container_width=True):
            if st.session_state.shares >= qty:
                st.session_state.balance += price * qty
                st.session_state.shares -= qty
                st.session_state.trades.append({"Type": "SELL", "Price": price, "Units": qty, "Time": datetime.now().strftime("%H:%M:%S")})
                st.rerun()

        total_value = st.session_state.balance + (st.session_state.shares * price)
        st.metric("Total Equity", f"${total_value:.2f}", delta=f"{(total_value-10000):.2f} PnL")

        if st.session_state.trades:
            st.table(pd.DataFrame(st.session_state.trades).tail(10))

else:
    st.warning("Please enter a valid ticker and ensure the market has enough data for calculations.")

