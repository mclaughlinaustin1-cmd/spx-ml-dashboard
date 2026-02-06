import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# --- Page Config ---
st.set_page_config(layout="wide", page_title="AI Terminal: Earnings & Volatility", page_icon="🏛️")

# --- Custom Styling ---
st.markdown("""
    <style>
    .main { background-color: #0b0e14; }
    .report-box { 
        background-color: #161b22; 
        border-left: 5px solid #00ffcc; 
        padding: 20px; 
        border-radius: 8px; 
        margin-bottom: 25px;
    }
    .status-up { color: #00ffcc; font-weight: bold; }
    .status-down { color: #ff4b4b; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- Core Logic ---

@st.cache_data(ttl=3600)
def fetch_serializable_data(ticker):
    """Fetches data and ensures all return types are serializable for Streamlit caching."""
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period="1y", interval="1d")
        if df.empty: return None, None
        
        # Flatten MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Extract Earnings Date
        earnings_date = None
        try:
            cal = tk.calendar
            if cal is not None and not cal.empty:
                earnings_date = cal.iloc[0, 0] if isinstance(cal, pd.DataFrame) else cal.get('Earnings Date', [None])[0]
        except: pass
            
        return df, earnings_date
    except:
        return None, None

def calculate_bollinger_bands(df, window=20, std_dev=2):
    df['BB_Mid'] = df['Close'].rolling(window=window).mean()
    df['BB_Std'] = df['Close'].rolling(window=window).std()
    df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * std_dev)
    df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * std_dev)
    return df

def get_ai_narrative(df, poly_coeffs, forecast_days, pred_end, earnings_date):
    current_price = df['Close'].iloc[-1]
    upper_band = df['BB_Upper'].iloc[-1]
    lower_band = df['BB_Lower'].iloc[-1]
    
    change_pct = ((pred_end - current_price) / current_price) * 100
    direction = "BULLISH" if pred_end > current_price else "BEARISH"
    
    # Volatility context
    vol_status = "STRETCHED (Upper Band)" if current_price > upper_band else \
                 "OVERSOLD (Lower Band)" if current_price < lower_band else "NEUTRAL"

    earnings_alert = ""
    if earnings_date:
        e_dt = pd.to_datetime(earnings_date).replace(tzinfo=None)
        days = (e_dt - datetime.now()).days
        if 0 <= days <= 14:
            earnings_alert = f"⚠️ **EARNINGS RISK:** Next report in {days} days. Expect high Aftermarket Gaps."

    return f"""
    ### 🤖 AI Market Narrative
    **Projected Move:** <span class="{'status-up' if direction == 'BULLISH' else 'status-down'}">{change_pct:+.2f}%</span> over {forecast_days} days.  
    **Volatility Context:** Currently **{vol_status}** relative to Bollinger Bands.  
    
    **Analysis:** The quadratic model plots a {direction.lower()} curve. Combined with recent {vol_status.lower()} positioning, 
    the model suggests a potential mean-reversion toward the $BB\_Mid$ ($${df['BB_Mid'].iloc[-1]:.2f}).
    
    {earnings_alert}
    """

# --- Sidebar ---
with st.sidebar:
    st.header("⚡ Terminal Config")
    ticker_str = st.text_input("Tickers (Separated by Commas)", "NVDA, TSLA, SPY")
    tickers = [t.strip().upper() for t in ticker_str.split(",")]
    
    st.divider()
    horizon = st.slider("Forecast Horizon (Days)", 1, 14, 7)
    st.info("Bollinger Bands: 20-Day SMA / 2x Std Dev")

# --- Main Terminal ---
st.title("Institutional Multi-Asset Terminal")

for symbol in tickers:
    df, next_earnings = fetch_serializable_data(symbol)
    
    if df is not None and len(df) > 20:
        df = calculate_bollinger_bands(df)
        df['Aftermarket_Shift'] = (df['Open'].shift(-1) - df['Close']) / df['Close'] * 100
        
        # AI Projection (Quadratic)
        y = df["Close"].values
        x = np.arange(len(y))
        coeffs = np.polyfit(x, y, 2)
        future_x = np.arange(len(y), len(y) + horizon)
        pred_y = np.polyval(coeffs, future_x)
        
        pred_dates = [df.index[-1] + timedelta(days=i+1) for i in range(horizon)]
        # Connect last real price to first prediction
        plot_dates = [df.index[-1]] + pred_dates
        plot_prices = [df["Close"].iloc[-1]] + list(pred_y)

        # UI Header & Narrative
        st.header(f"📊 {symbol} Analysis")
        narrative = get_ai_narrative(df, coeffs, horizon, plot_prices[-1], next_earnings)
        st.markdown(f'<div class="report-box">{narrative}</div>', unsafe_allow_html=True)

        # Plotting
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])

        # Candlesticks
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
        
        # Bollinger Bands
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='Upper Band', line=dict(color='rgba(173, 204, 255, 0.3)')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='Lower Band', line=dict(color='rgba(173, 204, 255, 0.3)'), fill='tonexty'), row=1, col=1)
        
        # AI Prediction
        fig.add_trace(go.Scatter(x=plot_dates, y=plot_prices, name='AI Forecast', line=dict(color='#00ffcc', width=3, dash='dashdot')), row=1, col=1)

        # Aftermarket Gaps
        gap_colors = ['#ff4b4b' if g < 0 else '#00ffcc' for g in df['Aftermarket_Shift']]
        fig.add_trace(go.Bar(x=df.index, y=df['Aftermarket_Shift'], name='Gap %', marker_color=gap_colors), row=2, col=1)

        fig.update_layout(template="plotly_dark", height=850, xaxis_rangeslider_visible=False, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        st.divider()
    else:
        st.warning(f"Insufficient data for {symbol}.")


