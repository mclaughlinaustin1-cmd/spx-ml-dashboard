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
TIMEFRAMES = {"1 Month": "1mo", "6 Months": "6mo", "1 Year": "1y", "Max": "max"}

@st.cache_data(ttl=3600)
def load_data(ticker, period):
    df = yf.download(ticker, period=period, interval="1d")
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
    
    # MACD & RSI
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + (gain / loss)))
    return df

# --- Sidebar UI ---
with st.sidebar:
    st.header("⚡ Terminal Settings")
    ticker = st.text_input("Asset Ticker", "NVDA").upper()
    tf = st.selectbox("Historical Range", list(TIMEFRAMES.keys()), index=0)
    chart_style = st.selectbox("Visualization", ["Candlesticks", "Hollow Line"])
    
    st.divider()
    st.subheader("🔮 Forecasting")
    show_forecast = st.toggle("Enable AI Prediction", True)
    forecast_days = st.slider("Forecast Horizon (Days)", 1, 14, 7)
    show_bb = st.toggle("Bollinger Analysis", True)

# --- Data Loading ---
df_raw = load_data(ticker, TIMEFRAMES[tf])

if df_raw is not None and len(df_raw) > 20:
    df = apply_indicators(df_raw.copy())
    
    # --- Prediction Logic ---
    # Using a 2nd degree polynomial (Quadratic) to predict curves rather than a flat line
    y = df["Close"].values
    x = np.arange(len(y))
    poly_model = np.polyfit(x, y, 2)
    
    # Generate future timestamps
    future_x = np.arange(len(y), len(y) + forecast_days)
    forecast_values = np.polyval(poly_model, future_x)
    
    last_date = df.index[-1]
    future_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
    
    # Create forecast Series for Plotly
    # We include the last known price to ensure the line is connected
    forecast_x = [df.index[-1]] + future_dates
    forecast_y = [df["Close"].iloc[-1]] + list(forecast_values)

    # --- Main Dashboard ---
    tab1, tab2 = st.tabs(["🏛 Institutional Market", "🧪 Algorithm Sandbox"])

    with tab1:
        m1, m2, m3, m4 = st.columns(4)
        curr_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        m1.metric("Live Price", f"${curr_price:.2f}", f"{(curr_price - prev_price):.2f}")
        m2.metric("RSI (14)", f"{df['RSI'].iloc[-1]:.1f}")
        m3.metric("Trend Velocity", f"{(forecast_y[-1] - curr_price):.2f}")
        m4.metric("Volatility", f"{(df['BB_High'].iloc[-1] - df['BB_Low'].iloc[-1]):.2f}")

        fig = go.Figure()

        # Bollinger Bands
        if show_bb:
            fig.add_trace(go.Scatter(x=df.index, y=df["BB_High"], name="Upper Band", line=dict(color='rgba(255,75,75,0.3)', width=1)))
            fig.add_trace(go.Scatter(x=df.index, y=df["BB_Low"], name="Lower Band", line=dict(color='rgba(0,255,204,0.3)', width=1), fill='tonexty', fillcolor='rgba(0, 255, 204, 0.02)'))

        # Price Plot
        if chart_style == "Candlesticks":
            fig.add_candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price")
        else:
            fig.add_trace(go.Scatter(x=df.index, y=df["Close"], line=dict(color='#ffffff', width=2), name="Close"))

        # VISIBLE DASHED PREDICTIVE LINE
        if show_forecast:
            fig.add_trace(go.Scatter(
                x=forecast_x, 
                y=forecast_y, 
                name="AI Prediction", 
                line=dict(color='#00ffcc', width=3, dash='dashdot'),
                mode='lines'
            ))

        fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False,
                          margin=dict(l=0, r=0, t=20, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

        st.line_chart(df["RSI"], height=150)

    with tab2:
        st.subheader("Risk Decision Logic")
        # Added a simple logic check based on the prediction
        if forecast_y[-1] > curr_price:
            st.success(f"PROJECTION: Bullish expansion expected over the next {forecast_days} days.")
        else:
            st.error(f"PROJECTION: Bearish correction signaled for the next {forecast_days} days.")

else:
    st.error("Please enter a valid ticker (e.g., AAPL, NVDA, BTC-USD).")
