import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

# --- Page Config ---
st.set_page_config(layout="wide", page_title="AI Alpha v9.8", page_icon="⚡")

# --- Core Data Engine (Optimized for Speed) ---
@st.cache_data(ttl=3600)
def get_terminal_data(ticker, horizon=5):
    df = yf.download(ticker, period="max", interval="1d")
    tnx = yf.download("^TNX", period="max", interval="1d")['Close']
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # 1. Macro & Momentum Features
    df['TNX'] = tnx
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_10'] = df['Log_Ret'].rolling(10).std()
    
    # 2. Bollinger Bands (Restored)
    # Mid = 20-period Moving Average
    # Upper/Lower = Mid +/- (2 * Standard Deviation)
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)
    
    # 3. RSI Calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    df['Target'] = df['Close'].shift(-horizon) / df['Close'] - 1
    return df.dropna()

# --- Sidebar ---
with st.sidebar:
    st.header("🏢 Terminal Controls")
    ticker = st.text_input("Ticker Symbol", "NVDA").upper()
    forecast_days = st.slider("Forecast Horizon (Days)", 1, 14, 5)
    st.divider()
    timeframes = {"1W": 7, "1M": 30, "3M": 90, "6M": 180, "1Y": 365, "2Y": 730}
    selected_tf = st.selectbox("Market Viewport", list(timeframes.keys()), index=2)
    lookback = timeframes[selected_tf]

data = get_terminal_data(ticker, forecast_days)

if data is not None:
    tab_live, tab_audit = st.tabs(["⚡ Live Prediction", "🧪 Advanced Audit"])

    with tab_live:
        # AI Inference Engine
        features = ['RSI', 'Vol_10', 'TNX']
        model = RandomForestRegressor(n_estimators=100, random_state=42).fit(data[features].values, data['Target'].values * 100)
        last_row = data.iloc[-1]
        pred_move = model.predict(last_row[features].values.reshape(1, -1))[0]
        
        # Signal Banner
        color = "#00ffcc" if pred_move > 0.5 else "#ff4b4b" if pred_move < -0.5 else "#ff9500"
        st.markdown(f"<div style='background-color:{color}22; border:2px solid {color}; padding:15px; border-radius:15px; text-align:center;'><h1>{ticker}: Target {pred_move:+.2f}%</h1></div>", unsafe_allow_html=True)

        vdf = data.tail(lookback)
        
        # --- 1. PRICE + BOLLINGER + FORECAST ---
        st.subheader("📈 Price Action & Bollinger Envelope")
        fig_p = go.Figure()
        
        # Bollinger Bands
        fig_p.add_trace(go.Scatter(x=vdf.index, y=vdf['BB_Upper'], line=dict(color='rgba(173, 216, 230, 0.2)'), showlegend=False))
        fig_p.add_trace(go.Scatter(x=vdf.index, y=vdf['BB_Lower'], line=dict(color='rgba(173, 216, 230, 0.2)'), fill='tonexty', fillcolor='rgba(173, 216, 230, 0.05)', name="Bollinger Bands"))
        
        # Candlesticks
        fig_p.add_trace(go.Candlestick(x=vdf.index, open=vdf['Open'], high=vdf['High'], low=vdf['Low'], close=vdf['Close'], name="History"))
        
        # Forecast Dotted Line
        future_date = vdf.index[-1] + timedelta(days=forecast_days)
        target_price = last_row['Close'] * (1 + (pred_move/100))
        fig_p.add_trace(go.Scatter(x=[vdf.index[-1], future_date], y=[last_row['Close'], target_price], line=dict(color=color, dash='dot', width=3), name="AI Target"))
        
        st.plotly_chart(fig_p.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False), use_container_width=True)
        st.info(f"**Interpretation:** We are viewing the last **{selected_tf}**. The **Bollinger Bands** show the volatility envelope; prices near the top band are 'stretched' high. The **Dotted Line** is our AI's destination for **${target_price:.2f}**.")

        # --- 2. VOLUME ---
        st.subheader("📊 Trading Activity (Volume)")
        v_colors = ['#00ffcc' if vdf['Close'].iloc[i] >= vdf['Open'].iloc[i] else '#ff4b4b' for i in range(len(vdf))]
        st.plotly_chart(go.Figure(go.Bar(x=vdf.index, y=vdf['Volume'], marker_color=v_colors)).update_layout(template="plotly_dark", height=200), use_container_width=True)
        st.warning(f"**Interpretation:** These bars track conviction over the **{selected_tf}** window. High bars confirm that the move is backed by real money.")

        # --- 3. MOMENTUM RSI (Orange Line) ---
        st.subheader("🏎️ Momentum Pulse (RSI)")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=vdf.index, y=vdf['RSI'], line=dict(color='orange', width=2), name="RSI"))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
        st.plotly_chart(fig_rsi.update_layout(template="plotly_dark", height=230, yaxis=dict(range=[0,100])), use_container_width=True)
        st.success(f"**Interpretation:** The **Orange Line** tracks momentum speed. Within this **{selected_tf}** view, a peak above the red line means the stock is 'Overbought'.")

    # --- TAB 2: AUDIT (Optimized for Speed) ---
    with tab_audit:
        st.subheader("🕵️ Performance Audit")
        audit_days = st.slider("Audit Window", 30, 730, 250)
        adf = data.tail(audit_days).copy()
        
        # High-Speed Vectorized Simulation
        adf['AI_Score'] = model.predict(adf[features].values)
        adf['Signal'] = np.where(adf['AI_Score'] > 0.5, 1, 0)
        adf['Strat_Ret'] = adf['Signal'].shift(1) * adf['Log_Ret']
        adf['Cum_Strat'] = np.exp(adf['Strat_Ret'].cumsum())
        adf['Cum_Mkt'] = np.exp(adf['Log_Ret'].cumsum())
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Alpha vs Market", f"{(adf['Cum_Strat'].iloc[-1] - adf['Cum_Mkt'].iloc[-1])*100:+.1f}%")
        m2.metric("Sharpe Ratio", f"{(adf['Strat_Ret'].mean() / adf['Strat_Ret'].std() * np.sqrt(252)):.2f}")
        m3.metric("Max Pain (Drawdown)", f"{((adf['Cum_Strat'] / adf['Cum_Strat'].cummax()) - 1).min() * 100:.1f}%")
        
        fig_audit = go.Figure()
        fig_audit.add_trace(go.Scatter(x=adf.index, y=adf['Cum_Strat'], name="AI Strategy", line=dict(color="#00ffcc")))
        fig_audit.add_trace(go.Scatter(x=adf.index, y=adf['Cum_Mkt'], name="Market", line=dict(color="gray", dash='dot')))
        st.plotly_chart(fig_audit.update_layout(template="plotly_dark", height=400), use_container_width=True)
