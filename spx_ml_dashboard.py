import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- Page Config ---
st.set_page_config(layout="wide", page_title="AI Institutional Terminal", page_icon="📈")

# --- Custom Styling ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div.stButton > button { width: 100%; border-radius: 5px; height: 3em; background-color: #262730; color: white; border: 1px solid #444; }
    div.stButton > button:hover { border: 1px solid #00ffcc; color: #00ffcc; }
    .metric-container { background-color: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- Helpers ---
TIMEFRAMES = {"1 Day": "1d", "1 Week": "5d", "1 Month": "1mo", "6 Months": "6mo", "1 Year": "1y", "Max": "max"}

@st.cache_data(ttl=3600)
def load_data(ticker, period):
    df = yf.download(ticker, period=period, interval="1h" if period in ["1d","5d"] else "1d")
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df

def apply_indicators(df):
    # Bollinger Bands
    df["MA20"] = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["BB_High"] = df["MA20"] + (std * 2)
    df["BB_Low"] = df["MA20"] - (std * 2)
    
    # Crossover Detection
    df['Cross_Up'] = np.where((df['Close'] < df['BB_Low']), df['Close'], np.nan)
    df['Cross_Down'] = np.where((df['Close'] > df['BB_High']), df['Close'], np.nan)
    
    # MACD
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    
    # RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + (gain / loss)))
    return df

# --- Sidebar UI ---
with st.sidebar:
    st.header("⚡ Terminal Settings")
    ticker = st.text_input("Asset Ticker", "AAPL").upper()
    tf = st.selectbox("Historical Range", list(TIMEFRAMES.keys()), index=2)
    chart_style = st.selectbox("Visualization", ["Candlesticks", "Hollow Line"])
    
    st.divider()
    show_forecast = st.toggle("AI Predictive Trend", True)
    show_bb = st.toggle("Bollinger Analysis", True)

# --- Data Loading ---
df_raw = load_data(ticker, TIMEFRAMES[tf])

if df_raw is not None and len(df_raw) > 20:
    df = apply_indicators(df_raw.copy())
    
    # --- Main Dashboard ---
    tab1, tab2 = st.tabs(["🏛 Institutional Market", "🧪 Algorithm Sandbox"])

    with tab1:
        # Metrics Header
        m1, m2, m3, m4 = st.columns(4)
        curr_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        m1.metric("Live Price", f"${curr_price:.2f}", f"{(curr_price - prev_price):.2f}")
        m2.metric("RSI (14)", f"{df['RSI'].iloc[-1]:.1f}")
        m3.metric("MACD Level", f"{df['MACD'].iloc[-1]:.3f}")
        m4.metric("Volatility (BB)", f"{(df['BB_High'].iloc[-1] - df['BB_Low'].iloc[-1]):.2f}")

        # Main Chart
        fig = go.Figure()

        # Bollinger Bands with distinct colors
        if show_bb:
            fig.add_trace(go.Scatter(x=df.index, y=df["BB_High"], name="Upper Band", line=dict(color='#ff4b4b', width=1)))
            fig.add_trace(go.Scatter(x=df.index, y=df["BB_Low"], name="Lower Band", line=dict(color='#00ffcc', width=1), fill='tonexty', fillcolor='rgba(0, 255, 204, 0.03)'))
            
            # Data Point Markers for Crossovers
            fig.add_trace(go.Scatter(x=df.index, y=df['Cross_Up'], mode='markers', name='Oversold (Buy)',
                                     marker=dict(symbol='triangle-up', size=10, color='#00ffcc', line=dict(width=1, color='white'))))
            fig.add_trace(go.Scatter(x=df.index, y=df['Cross_Down'], mode='markers', name='Overbought (Sell)',
                                     marker=dict(symbol='triangle-down', size=10, color='#ff4b4b', line=dict(width=1, color='white'))))

        # Price Plot
        if chart_style == "Candlesticks":
            fig.add_candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price")
        else:
            fig.add_trace(go.Scatter(x=df.index, y=df["Close"], line=dict(color='#ffffff', width=2), name="Close"))

        fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False,
                          margin=dict(l=0, r=0, t=20, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

        # Secondary indicators
        col_a, col_b = st.columns(2)
        with col_a:
            st.caption("Relative Strength Index")
            st.line_chart(df["RSI"], height=200)
        with col_b:
            st.caption("MACD Impulse")
            st.line_chart(df[["MACD", "Signal"]], height=200)

    with tab2:
        # Logic for Trading Simulator
        if "cash" not in st.session_state:
            st.session_state.update({"cash": 10000.0, "inv": 0, "log": []})

        st.subheader("💰 Active Paper Portfolio")
        s1, s2, s3 = st.columns(3)
        s1.metric("Cash Balance", f"${st.session_state.cash:.2f}")
        s2.metric("Asset Holdings", f"{st.session_state.inv} Units")
        
        trade_qty = st.number_input("Order Quantity", 1, 1000, 10)
        c_buy, c_sell = st.columns(2)
        
        if c_buy.button("EXECUTE BUY"):
            cost = trade_qty * curr_price
            if st.session_state.cash >= cost:
                st.session_state.cash -= cost
                st.session_state.inv += trade_qty
                st.session_state.log.append(f"BUY {trade_qty} {ticker} @ {curr_price:.2f}")
                st.rerun()
        
        if c_sell.button("EXECUTE SELL"):
            if st.session_state.inv >= trade_qty:
                st.session_state.cash += trade_qty * curr_price
                st.session_state.inv -= trade_qty
                st.session_state.log.append(f"SELL {trade_qty} {ticker} @ {curr_price:.2f}")
                st.rerun()
        
        if st.session_state.log:
            st.write("Recent Activity")
            st.code("\n".join(st.session_state.log[-5:]))

else:
    st.error("Terminal offline. Please verify Ticker Symbol and connection.")


