import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Institutional AI Multi-Terminal", page_icon="🏦")

# --- Custom Styling ---
st.markdown("""
    <style>
    .main { background-color: #0b0e14; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; padding: 10px; border-radius: 8px; }
    .report-box { 
        background-color: #161b22; 
        border-left: 5px solid #00ffcc; 
        padding: 20px; 
        border-radius: 5px; 
        margin-bottom: 25px;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    .status-up { color: #00ffcc; font-weight: bold; }
    .status-down { color: #ff4b4b; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- AI Logic Modules ---

def get_ai_narrative(poly_coeffs, forecast_days, current_price, end_price):
    """Translates quadratic math into a human-readable trend report."""
    a, b, c = poly_coeffs
    direction = "BULLISH" if end_price > current_price else "BEARISH"
    diff = end_price - current_price
    change_pct = (diff / current_price) * 100
    
    # Interpretation of curvature (a coefficient)
    if abs(a) < 0.001:
        momentum_desc = "maintaining a steady, linear trajectory."
    elif a > 0:
        momentum_desc = "displaying exponential acceleration (the curve is steepening upwards)."
    else:
        momentum_desc = "showing signs of momentum exhaustion (the curve is rounding off)."

    return f"""
    ### 🧠 AI Projection Narrative
    For the next **{forecast_days} days**, the model predicts a <span class="{'status-up' if direction == 'BULLISH' else 'status-down'}">{direction}</span> movement.
    
    **Analysis:** The price is currently {momentum_desc} Based on current velocity, the AI projects a price target of **${end_price:.2f}**, representing a **{change_pct:+.2f}%** shift from the current close.
    
    *Note: This is a 2nd-degree polynomial regression based on historical volatility.*
    """

@st.cache_data(ttl=3600)
def fetch_data(ticker):
    try:
        df = yf.download(ticker, period="1y", interval="1d")
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.dropna(inplace=True)
        return df
    except Exception:
        return None

def apply_indicators(df, whale_threshold):
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
    
    # Whale Detection
    df["Whale_Val"] = df["Volume"] * df["Close"]
    df["Whale_Alert"] = df["Whale_Val"] > whale_threshold
    return df

# --- Sidebar UI ---
with st.sidebar:
    st.header("⚙️ Terminal Control")
    ticker_input = st.text_input("Enter Tickers (Comma Separated)", "AAPL, NVDA, BTC-USD")
    tickers = [t.strip().upper() for t in ticker_input.split(",")]
    
    st.divider()
    forecast_days = st.slider("Prediction Horizon (Days)", 1, 14, 7)
    whale_limit = st.number_input("Whale Tx Threshold ($)", value=10000000, step=1000000)
    
    st.divider()
    st.caption("Hover over charts to sync data points across all indicators.")

# --- Main Terminal Loop ---
st.title("Institutional Multi-Ticker Terminal")

for symbol in tickers:
    df_raw = fetch_data(symbol)
    
    if df_raw is not None and len(df_raw) > 30:
        df = apply_indicators(df_raw.copy(), whale_limit)
        
        # --- AI Forecasting Logic ---
        y = df["Close"].values
        x = np.arange(len(y))
        poly_coeffs = np.polyfit(x, y, 2)
        
        future_x = np.arange(len(y), len(y) + forecast_days)
        prediction = np.polyval(poly_coeffs, future_x)
        
        # Dates for plotting
        pred_dates = [df.index[-1]] + [df.index[-1] + timedelta(days=i+1) for i in range(forecast_days)]
        pred_prices = [df["Close"].iloc[-1]] + list(prediction)
        
        # --- Asset Layout ---
        st.header(f"📊 Market Analysis: {symbol}")
        
        # 1. AI Narrative Box
        narrative_html = get_ai_narrative(poly_coeffs, forecast_days, df["Close"].iloc[-1], pred_prices[-1])
        st.markdown(f'<div class="report-box">{narrative_html}</div>', unsafe_allow_html=True)
        
        # 2. Synchronized Charts (Subplots)
        fig = make_subplots(
            rows=3, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.02, 
            row_heights=[0.5, 0.25, 0.25]
        )

        # Main Price & Prediction
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=pred_dates, y=pred_prices, name='AI Forecast', 
            line=dict(color='#00ffcc', width=3, dash='dashdot')
        ), row=1, col=1)

        # Whale Markers
        whales = df[df["Whale_Alert"]]
        fig.add_trace(go.Scatter(
            x=whales.index, y=whales['Close'], mode='markers', name='Whale Activity',
            marker=dict(color='gold', size=10, symbol='diamond')
        ), row=1, col=1)

        # RSI
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='#ff4b4b')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="gray", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="gray", row=2, col=1)

        # MACD
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='#00d1ff')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], name='Signal', line=dict(color='#ff9500')), row=3, col=1)

        # Global Chart Settings
        fig.update_layout(
            template="plotly_dark", height=850, 
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            margin=dict(l=0, r=0, t=20, b=0)
        )
        
        # Range Selectors on bottom X-Axis
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
            row=3, col=1
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # 3. Educational Definitions
        with st.expander(f"📚 Indicator Glossary for {symbol}"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**RSI (Relative Strength Index)**")
                st.write("Measures momentum. Above 70 suggests overbought (potential sell), below 30 suggests oversold (potential buy).")
            with c2:
                st.markdown("**MACD (Trend Strength)**")
                st.write("When the Blue line crosses above the Orange line, a bullish trend is likely starting.")
        
        st.divider()
    else:
        st.error(f"Data for {symbol} is unavailable. Please check the ticker.")

