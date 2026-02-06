import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

# --- Core Model Logic with Technical Indicators ---
@st.cache_data(ttl=3600)
def build_quant_data(ticker):
    df = yf.download(ticker, period="5y", interval="1d")
    if df.empty: return None
    
    # Fix MultiIndex for recent yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Log Returns for Stationarity
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Bollinger Bands Calculation (20-day SMA +/- 2 Std Dev)
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['20STD'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['MA20'] + (df['20STD'] * 2)
    df['BB_Lower'] = df['MA20'] - (df['20STD'] * 2)
    
    # RSI Calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    # ML Prep
    df['Vol_10'] = df['Log_Ret'].rolling(10).std()
    df = df.dropna()
    
    features = ['RSI', 'Vol_10', 'Log_Ret']
    X = df[features].values
    y = df['Log_Ret'].values
    
    scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X)
    
    split = len(df) - 100
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_s[:split], y[:split])
    
    # Win Rate Calculation
    preds = model.predict(X_s[split:])
    win_rate = (np.sign(preds) == np.sign(y[split:])).mean() * 100
    
    return df, model, win_rate, scaler

# --- UI Layout ---
st.set_page_config(layout="wide", page_title="AI Quant Terminal v4.2")
st.title("🏛️ AI Quant Terminal: Bollinger Strategy")

ticker = st.sidebar.text_input("Enter Ticker", "NVDA").upper()
target_dt = st.sidebar.date_input("Target Date", datetime.now() + timedelta(days=10))

data = build_quant_data(ticker)

if data:
    df, model, win_rate, scaler = data
    
    # --- Prediction Execution ---
    days_out = (pd.Timestamp(target_dt) - df.index[-1]).days
    last_state = scaler.transform(df[['RSI', 'Vol_10', 'Log_Ret']].tail(1).values)
    pred_ret = model.predict(last_state)[0]
    final_price = df['Close'].iloc[-1] * np.exp(pred_ret * days_out)

    # --- Candlestick Chart + Bollinger Bands ---
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    
    # Price and Candles
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
    
    # Bollinger Bands Overlay
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='rgba(173, 204, 255, 0.3)', width=1), name="BB Upper"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='rgba(173, 204, 255, 0.3)', width=1), fill='tonexty', fillcolor='rgba(173, 204, 255, 0.1)', name="BB Lower"), row=1, col=1)
    
    # Prediction Visualization
    fig.add_trace(go.Scatter(x=[df.index[-1], pd.Timestamp(target_dt)], y=[df['Close'].iloc[-1], final_price], line=dict(color='#ff9500', width=4, dash='dot'), name="AI Prediction"), row=1, col=1)

    # RSI Subplot
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#ff9500'), name="RSI"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    fig.update_layout(template="plotly_dark", height=800, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- Summary Metrics & Briefing ---
    st.subheader(f"📊 Market Intelligence Report")
    c1, c2, c3 = st.columns(3)
    c1.metric("Target Price", f"${final_price:,.2f}")
    c2.metric("Directional Win-Rate", f"{win_rate:.1f}%")
    c3.metric("BB Bandwidth", f"{((df['BB_Upper'].iloc[-1] - df['BB_Lower'].iloc[-1]) / df['MA20'].iloc[-1] * 100):.2f}%")

    st.info(f"**Brief:** The model has analyzed the last 5 years of {ticker} history. With a historical win-rate of **{win_rate:.1f}%**, it predicts the stock will reach **${final_price:,.2f}** by {target_dt}. The price is currently {'near the upper band' if df['Close'].iloc[-1] > df['BB_Upper'].iloc[-1] * 0.95 else 'trading within normal range'}.")

else:
    st.error("Ticker not found or data connection error.")

