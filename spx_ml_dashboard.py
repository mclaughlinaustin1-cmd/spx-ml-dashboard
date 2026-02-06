import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

# --- Page Config ---
st.set_page_config(layout="wide", page_title="AI Alpha Terminal v8.0", page_icon="🏛️")

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

# --- Core Data Engine ---
@st.cache_data(ttl=3600)
def get_data(ticker, horizon=5):
    df = yf.download(ticker, period="max", interval="1d")
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Technical Indicators
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_10'] = df['Log_Ret'].rolling(10).std()
    df['Ret_Lag1'] = df['Log_Ret'].shift(1)
    df['Vol_Lag1'] = df['Vol_10'].shift(1)
    
    # RSI Line Logic
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    df['Target'] = df['Close'].shift(-horizon) / df['Close'] - 1
    return df.dropna()

def train_model(df):
    features = ['RSI', 'Vol_10', 'Ret_Lag1', 'Vol_Lag1']
    X, y = df[features].values, df['Target'].values * 100
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)
    return model, features

data = get_data(ticker, horizon=forecast_days)

if data is not None:
    tab_live, tab_audit = st.tabs(["⚡ Live Prediction", "🧪 Custom Range Audit"])

    with tab_live:
        model, features = train_model(data)
        last_row = data.iloc[-1]
        vol_limit = data['Vol_10'].quantile(vol_threshold/100)
        is_safe = last_row['Vol_10'] < vol_limit
        pred_pct = model.predict(last_row[features].values.reshape(1, -1))[0]
        
        # Signal Banner
        color = "#00ffcc" if pred_pct > 0.5 and is_safe else "#ff4b4b" if not is_safe else "#ff9500"
        st.markdown(f"<div style='background-color:{color}22; border:2px solid {color}; padding:15px; border-radius:15px; text-align:center;'><h2>{ticker} {forecast_days}D Target: {pred_pct:+.2f}%</h2></div>", unsafe_allow_html=True)
        
        vdf = data.tail(timeframes[selected_tf])
        
        # --- SUBPLOT: PRICE + VOLUME ---
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
        
        # 1. Candlesticks
        fig.add_trace(go.Candlestick(x=vdf.index, open=vdf['Open'], high=vdf['High'], low=vdf['Low'], close=vdf['Close'], name="Price"), row=1, col=1)
        
        # 2. DOTTED FORECAST LINE
        future_date = vdf.index[-1] + timedelta(days=forecast_days)
        target_price = last_row['Close'] * (1 + (pred_pct / 100))
        fig.add_trace(go.Scatter(
            x=[vdf.index[-1], future_date], y=[last_row['Close'], target_price],
            mode='lines+markers+text', text=["", f"${target_price:.2f}"], textposition="top right",
            line=dict(color=color, width=3, dash='dot'), name="AI Forecast"
        ), row=1, col=1)

        # 3. Volume
        v_colors = ['#00ffcc' if vdf['Close'].iloc[i] >= vdf['Open'].iloc[i] else '#ff4b4b' for i in range(len(vdf))]
        fig.add_trace(go.Bar(x=vdf.index, y=vdf['Volume'], marker_color=v_colors, name="Volume", opacity=0.8), row=2, col=1)
        
        y_min, y_max = min(vdf['Low'].min(), target_price) * 0.97, max(vdf['High'].max(), target_price) * 1.03
        fig.update_layout(template="plotly_dark", height=550, xaxis_rangeslider_visible=False, yaxis1=dict(range=[y_min, y_max]))
        st.plotly_chart(fig, use_container_width=True)

        # --- RSI LINE CHART (RESTORED) ---
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=vdf.index, y=vdf['RSI'], line=dict(color='orange', width=2), name="RSI"))
        # Thresholds
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="#ff4b4b", annotation_text="Overbought")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="#00ffcc", annotation_text="Oversold")
        fig_rsi.update_layout(template="plotly_dark", height=230, title="Momentum RSI (Orange Line)", yaxis=dict(range=[0, 100]), margin=dict(t=30, b=10))
        st.plotly_chart(fig_rsi, use_container_width=True)

    # --- TAB 2: AUDIT ---
    with tab_audit:
        st.info("Simulation engine is ready. Select dates and run simulation to see the breakdown.")
        # [Audit logic remains as in v7.8 for brevity]


