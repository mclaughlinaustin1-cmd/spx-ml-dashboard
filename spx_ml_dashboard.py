import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

# --- Core Setup ---
st.set_page_config(layout="wide", page_title="AI Alpha Terminal v5.3")
ticker = st.sidebar.text_input("Global Ticker", "AAPL").upper()

# Create the Tabs
tab_live, tab_audit = st.tabs(["🏛️ Live Terminal", "🕵️ Historical Audit"])

@st.cache_data(ttl=3600)
def get_clean_data(ticker):
    df = yf.download(ticker, period="max", interval="1d")
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Feature Engineering
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['MA20'] = df['Close'].rolling(20).mean()
    df['STD20'] = df['Close'].rolling(20).std()
    df['BB_Up'] = df['MA20'] + (df['STD20'] * 2)
    df['BB_Low'] = df['MA20'] - (df['STD20'] * 2)
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    df['Vol_10'] = df['Log_Ret'].rolling(10).std()
    return df.dropna()

data_full = get_clean_data(ticker)

# --- TAB 1: LIVE TERMINAL ---
with tab_live:
    if data_full is not None:
        st.subheader(f"Current Analysis for {ticker}")
        
        # Train on all available data for the live forecast
        features = ['RSI', 'Vol_10', 'Log_Ret']
        scaler = StandardScaler().fit(data_full[features])
        X_s = scaler.transform(data_full[features])
        
        model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_s, data_full['Log_Ret'])
        
        # Signal Logic (simplified for brevity)
        last_row = data_full.iloc[-1]
        pred_7d = model.predict(X_s[-1:]) [0]
        target = last_row['Close'] * np.exp(pred_7d * 7)
        
        # Decision UI
        sig = "BUY" if (last_row['RSI'] < 35 or last_row['Close'] < last_row['BB_Low']) else "SELL" if (last_row['RSI'] > 65 or last_row['Close'] > last_row['BB_Up']) else "HOLD"
        color = "#00ffcc" if sig == "BUY" else "#ff4b4b" if sig == "SELL" else "#ff9500"
        
        st.markdown(f"<div style='border:2px solid {color}; padding:20px; border-radius:10px; text-align:center;'><h1 style='color:{color};'>{sig}</h1><p>7-Day Target: ${target:,.2f}</p></div>", unsafe_allow_html=True)
        
        # Chart
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Candlestick(x=data_full.index[-100:], open=data_full['Open'], high=data_full['High'], low=data_full['Low'], close=data_full['Close']))
        fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: HISTORICAL AUDIT ---
with tab_audit:
    st.subheader("🕵️ Backtest on Custom Dates")
    st.write("Pick a date range in the past to see how the model *would* have performed.")
    
    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start Date", datetime.now() - timedelta(days=365))
    end_date = col2.date_input("End Date", datetime.now() - timedelta(days=30))
    
    if st.button("Run Historical Audit"):
        # Filter data up to the end_date for realistic testing
        audit_data = data_full[:pd.Timestamp(end_date)]
        test_window = audit_data[pd.Timestamp(start_date):]
        train_window = data_full[:pd.Timestamp(start_date)]
        
        if len(train_window) > 100:
            # 1. Train only on PRE-START data (No peeking!)
            scaler_audit = StandardScaler().fit(train_window[features])
            X_train = scaler_audit.transform(train_window[features])
            model_audit = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
            model_audit.fit(X_train, train_window['Log_Ret'])
            
            # 2. Test on the selected window
            X_test = scaler_audit.transform(test_window[features])
            preds = model_audit.predict(X_test)
            
            # Calculate Accuracy
            actuals = test_window['Log_Ret'].values
            hits = (np.sign(preds) == np.sign(actuals)) & (np.abs(actuals) > 0.005)
            accuracy = hits.mean() * 100
            
            # 3. Display Results
            st.metric("Audit Accuracy Score", f"{accuracy:.1f}%")
            
            # Equity Curve for that specific window
            test_window = test_window.copy()
            test_window['Strategy_Ret'] = np.sign(preds) * actuals
            test_window['Equity'] = (1 + test_window['Strategy_Ret']).cumprod() * 10000
            
            fig_audit = go.Figure()
            fig_audit.add_trace(go.Scatter(x=test_window.index, y=test_window['Equity'], name="AI Strategy P/L", line=dict(color="#00ffcc")))
            fig_audit.update_layout(title=f"Growth of $10k during {start_date} to {end_date}", template="plotly_dark")
            st.plotly_chart(fig_audit, use_container_width=True)
        else:
            st.error("Not enough historical data before start date to train the model!")

