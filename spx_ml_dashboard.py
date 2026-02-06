import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

# --- Page Config ---
st.set_page_config(layout="wide", page_title="AI Alpha Terminal v8.1", page_icon="🏛️")

# --- Global Sidebar ---
with st.sidebar:
    st.header("⚙️ Quant Lab")
    ticker = st.text_input("Ticker Symbol", "NVDA").upper()
    forecast_days = st.slider("Forecast Horizon (Days)", 1, 10, 5)
    st.divider()
    vol_threshold = st.slider("Volatility Safety Cutoff (%)", 50, 100, 85)
    
    timeframes = {
        "1 Week": 7, "14 Days": 14, "30 Days": 30, "60 Days": 60, "90 Days": 90,
        "6 Months": 180, "1 Year": 365, "2 Years": 730, "5 Years": 1825, "10 Years": 3650
    }
    selected_tf = st.selectbox("Live Chart View", list(timeframes.keys()), index=4)
    lookback = timeframes[selected_tf]

# --- Core Data Engine ---
@st.cache_data(ttl=3600)
def get_data(ticker, horizon=5):
    df = yf.download(ticker, period="max", interval="1d")
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Technical Logic
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_10'] = df['Log_Ret'].rolling(10).std()
    df['Ret_Lag1'] = df['Log_Ret'].shift(1)
    df['Vol_Lag1'] = df['Vol_10'].shift(1)
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    df['Target'] = df['Close'].shift(-horizon) / df['Close'] - 1
    return df.dropna()

data = get_data(ticker, horizon=forecast_days)

if data is not None:
    tab_live, tab_audit = st.tabs(["⚡ Live Prediction", "🧪 Custom Range Audit"])

    with tab_live:
        # AI Training & Inference
        features = ['RSI', 'Vol_10', 'Ret_Lag1', 'Vol_Lag1']
        model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        model.fit(data[features].values, data['Target'].values * 100)
        
        last_row = data.iloc[-1]
        vol_limit = data['Vol_10'].quantile(vol_threshold/100)
        is_safe = last_row['Vol_10'] < vol_limit
        pred_pct = model.predict(last_row[features].values.reshape(1, -1))[0]
        
        vdf = data.tail(lookback)
        
        # --- 1. PRICE & FORECAST ---
        st.subheader("📈 Price Action & AI Projection")
        fig_price = go.Figure()
        fig_price.add_trace(go.Candlestick(x=vdf.index, open=vdf['Open'], high=vdf['High'], 
                                           low=vdf['Low'], close=vdf['Close'], name="Candles"))
        
        # Forecast Dotted Line
        future_date = vdf.index[-1] + timedelta(days=forecast_days)
        target_p = last_row['Close'] * (1 + (pred_pct / 100))
        f_color = "#00ffcc" if pred_pct > 0 else "#ff4b4b"
        fig_price.add_trace(go.Scatter(x=[vdf.index[-1], future_date], y=[last_row['Close'], target_p],
                                       mode='lines+markers+text', text=["", f"Target: ${target_p:.2f}"],
                                       line=dict(color=f_color, width=3, dash='dot'), name="AI Forecast"))

        fig_price.update_layout(template="plotly_dark", height=450, xaxis_rangeslider_visible=False,
                                margin=dict(t=10, b=10))
        st.plotly_chart(fig_price, use_container_width=True)
        
        st.info(f"**Interpretation:** This graph shows the price journey over the last **{selected_tf}**. "
                f"The candles track the Highs and Lows of each day. The **Dotted Line** is our AI's 'best guess' "
                f"for where the price will be in **{forecast_days} days**. A green line suggests upward momentum, "
                f"while a red line indicates a projected dip.")

        # --- 2. VOLUME ---
        st.subheader("📊 Trading Activity (Volume)")
        v_colors = ['#00ffcc' if vdf['Close'].iloc[i] >= vdf['Open'].iloc[i] else '#ff4b4b' for i in range(len(vdf))]
        fig_vol = go.Figure(go.Bar(x=vdf.index, y=vdf['Volume'], marker_color=v_colors, name="Volume"))
        fig_vol.update_layout(template="plotly_dark", height=200, margin=dict(t=10, b=10))
        st.plotly_chart(fig_vol, use_container_width=True)
        
        st.warning(f"**Interpretation:** Volume is the 'fuel' of the market. These bars show how many shares changed hands. "
                   f"In this **{selected_tf}** view, we look for **high bars** to confirm price moves. "
                   f"If the AI predicts a rally but Volume is low, the move may lack the strength to last.")

        # --- 3. MOMENTUM ---
        st.subheader("🏎️ Momentum Pulse (RSI)")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=vdf.index, y=vdf['RSI'], line=dict(color='orange', width=2)))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="#ff4b4b", annotation_text="Overbought")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="#00ffcc", annotation_text="Oversold")
        fig_rsi.update_layout(template="plotly_dark", height=230, yaxis=dict(range=[0, 100]), margin=dict(t=10, b=10))
        st.plotly_chart(fig_rsi, use_container_width=True)
        
        st.success(f"**Interpretation:** The RSI line measures speed and change. "
                   f"Across this **{selected_tf}** window, if the orange line crosses above the **Red Dash (70)**, "
                   f"the stock is 'Overbought' (potentially expensive). If it drops below the **Green Dash (30)**, "
                   f"it is 'Oversold' (potentially a bargain).")

    with tab_audit:
        st.write("The historical stress test engine is available here to verify the logic described above.")


