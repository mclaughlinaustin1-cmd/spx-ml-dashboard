import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# --- Page Config ---
st.set_page_config(layout="wide", page_title="AI Institutional Terminal", page_icon="🏛️")

# --- Custom UI Styling ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; padding: 10px; border-radius: 8px; }
    [data-testid="stExpander"] { border: 1px solid #30363d; background: #161b22; }
    </style>
    """, unsafe_allow_html=True)

# --- Logic Modules ---

@st.cache_data(ttl=3600)
def fetch_terminal_data(ticker):
    tk = yf.Ticker(ticker)
    df = tk.history(period="1y")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df, tk.news

def process_indicators(df, whale_threshold):
    # Bollinger Bands
    df["MA20"] = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["BB_High"] = df["MA20"] + (std * 2)
    df["BB_Low"] = df["MA20"] - (std * 2)
    
    # RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + (gain / loss)))
    
    # MACD
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    
    # Whale Alerts ($Val = Vol * Price)
    df["Whale_Val"] = df["Volume"] * df["Close"]
    df["Whale_Alert"] = df["Whale_Val"] > whale_threshold
    
    return df

# --- Sidebar UI ---
with st.sidebar:
    st.header("⚡ Terminal Control")
    ticker = st.text_input("Asset Ticker", "NVDA").upper()
    forecast_horizon = st.slider("Prediction Horizon (Days)", 1, 14, 7)
    whale_limit = st.number_input("Whale Threshold ($)", value=5000000, step=1000000)
    st.divider()
    st.info("Crosshair Sync: Hover over any point on the main chart to see synced data across indicators.")

# --- Main Execution ---
df_raw, news = fetch_terminal_data(ticker)

if not df_raw.empty:
    df = process_indicators(df_raw.copy(), whale_limit)
    
    # --- 14-Day Prediction Logic ---
    y_vals = df["Close"].values
    x_vals = np.arange(len(y_vals))
    poly_fit = np.polyfit(x_vals, y_vals, 2)
    future_x = np.arange(len(y_vals), len(y_vals) + forecast_horizon)
    prediction = np.polyval(poly_fit, future_x)
    
    # Connector for smooth line
    pred_x = [df.index[-1]] + [df.index[-1] + timedelta(days=i+1) for i in range(forecast_horizon)]
    pred_y = [df["Close"].iloc[-1]] + list(prediction)

    # --- MASTER CHART (Subplots) ---
    # Creating 3 rows: Price (0.6), RSI (0.2), MACD (0.2)
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05, 
        row_heights=[0.6, 0.2, 0.2]
    )

    # 1. Main Price Trace
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'
    ), row=1, col=1)

    # 2. Whale Markers
    whales = df[df["Whale_Alert"]]
    fig.add_trace(go.Scatter(
        x=whales.index, y=whales['Close'], mode='markers', name='Whale Tx > $5M',
        marker=dict(color='gold', size=10, symbol='diamond', line=dict(color='white', width=1))
    ), row=1, col=1)

    # 3. AI Predictive Line (Dashed)
    fig.add_trace(go.Scatter(
        x=pred_x, y=pred_y, name='AI Prediction', 
        line=dict(color='#00ffcc', width=3, dash='dashdot')
    ), row=1, col=1)

    # 4. RSI Indicator
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='#ff4b4b')), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="gray", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="gray", row=2, col=1)

    # 5. MACD Indicator
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='#00d1ff')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], name='Signal', line=dict(color='#ff9500')), row=3, col=1)

    # Formatting UI for manipulation
    fig.update_layout(
        template="plotly_dark", height=900, 
        xaxis_rangeslider_visible=False,
        hovermode="x unified", # SYNCHRONIZED CROSSHAIR
        margin=dict(l=10, r=10, t=30, b=10)
    )
    
    # Range Selector UI
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(step="all")
            ]),
            bgcolor="#161b22"
        ),
        row=3, col=1 # Put slider at the very bottom
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- EDUCATIONAL SECTION ---
    st.divider()
    col_a, col_b = st.columns(2)
    
    with col_a:
        with st.expander("📊 RSI Logic (Relative Strength Index)"):
            st.write("""
            **What it shows:** Momentum speed. 
            - **Overbought (>70):** Price may be stretched too high; correction likely.
            - **Oversold (<30):** Price may be beaten down too far; recovery likely.
            """)
    with col_b:
        with st.expander("📈 MACD Logic (Momentum Divergence)"):
            st.write("""
            **What it shows:** Trend strength.
            - **MACD above Signal (Blue over Orange):** Bullish momentum.
            - **MACD below Signal:** Bearish momentum.
            """)

else:
    st.error("Terminal Offline. Check Ticker connection.")

