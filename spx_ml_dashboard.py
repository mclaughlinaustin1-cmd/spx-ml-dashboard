import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime, timedelta

# --- Page Config ---
st.set_page_config(layout="wide", page_title="AI Alpha Terminal v6.1", page_icon="🏛️")

# --- Global Sidebar ---
with st.sidebar:
    st.header("⚙️ Terminal Config")
    ticker = st.text_input("Ticker Symbol", "NVDA").upper()
    target_dt = st.date_input("Forecast Target Date", datetime.now() + timedelta(days=7))
    st.divider()
    st.caption("v6.1: Logic Decoder & Drawdown Analysis")

@st.cache_data(ttl=3600)
def get_terminal_data(ticker):
    df = yf.download(ticker, period="10y", interval="1d")
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
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
    df = df.dropna()
    
    features = ['RSI', 'Vol_10', 'Log_Ret']
    X = df[features].values
    y = df['Log_Ret'].values
    scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X)
    
    tscv = TimeSeriesSplit(n_splits=5)
    wr_scores = []
    for tr_i, te_i in tscv.split(X_s):
        m = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        m.fit(X_s[tr_i], y[tr_i])
        p = m.predict(X_s[te_i])
        w = (np.sign(p) == np.sign(y[te_i])) & (np.abs(y[te_i]) >= 0.005)
        wr_scores.append(w.mean())
    
    final_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42).fit(X_s, y)
    return df, final_model, np.mean(wr_scores) * 100, scaler

data_package = get_terminal_data(ticker)

if data_package:
    df, model, win_rate, scaler = data_package
    tab_live, tab_audit = st.tabs(["🏛️ Live Terminal", "🕵️ Historical Audit"])

    with tab_live:
        # Prediction Logic
        last_s = scaler.transform(df[['RSI', 'Vol_10', 'Log_Ret']].tail(1).values)
        p_ret = model.predict(last_s)[0]
        days_out = (pd.Timestamp(target_dt) - df.index[-1]).days
        final_p = df['Close'].iloc[-1] * np.exp(p_ret * max(1, days_out))
        proj_m = ((final_p / df['Close'].iloc[-1]) - 1) * 100
        
        # Decision Banner
        decision = "BUY" if proj_m > 0.5 else "SELL" if proj_m < -0.5 else "HOLD"
        color = "#00ffcc" if decision == "BUY" else "#ff4b4b" if decision == "SELL" else "#ff9500"
        st.markdown(f"<div style='background-color:{color}22; border:2px solid {color}; padding:20px; border-radius:12px; text-align:center;'><h1>SIGNAL: {decision}</h1></div>", unsafe_allow_html=True)
        
        # Charts
        c1, c2 = st.columns([3, 1])
        with c1:
            fig_p = go.Figure()
            fig_p.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
            fig_p.update_layout(template="plotly_dark", height=450, xaxis_range=[df.index[-60], df.index[-1]])
            st.plotly_chart(fig_p, use_container_width=True)
        with c2:
            st.metric("Model Confidence", f"{win_rate:.1f}%")
            st.metric("Target Price", f"${final_p:,.2f}")
        
        # --- NEW: AI LOGIC DECODER ---
        st.divider()
        st.subheader("🧠 How the AI is Predicting for You")
        clogic1, clogic2 = st.columns(2)
        with clogic1:
            st.markdown("""
            **The 'Random Forest' Brain:**
            The model isn't just looking at a line; it's a "forest" of 100 different decision trees. Each tree votes on the most likely outcome based on three specific inputs:
            1. **Momentum (RSI):** Is the stock over-extended or primed for a bounce?
            2. **Market Stress (Volatility):** How erratic has the price been in the last 10 days?
            3. **History (Log Returns):** Does the current pattern look like previous 'winning' patterns in the last 10 years?
            """)
        with clogic2:
            st.markdown(f"""
            **Current Weighting Analysis:**
            * The model identified a **{proj_m:.2f}%** expected move.
            * It uses **Non-Linear logic**, meaning it knows that an RSI of 70 in a high-volatility market means something different than an RSI of 70 in a calm market.
            * **Final Verdict:** The target of **${final_p:,.2f}** is the average 'consensus' of those 100 internal trees.
            """)

    with tab_audit:
        st.subheader("🕵️ Performance Stress Test")
        min_d, max_d = df.index.min().date(), df.index.max().date()
        ca, cb = st.columns(2)
        audit_start = ca.date_input("Audit Start Date", value=max_d - timedelta(days=365), min_value=min_d + timedelta(days=150))
        audit_end = cb.date_input("Audit End Date", value=max_d)
        
        if st.button("🚀 Run Backtest"):
            t_start, t_end = pd.Timestamp(audit_start), pd.Timestamp(audit_end)
            train_box = df[df.index < t_start]
            test_box = df[(df.index >= t_start) & (df.index <= t_end)].copy()
            
            # Training
            features = ['RSI', 'Vol_10', 'Log_Ret']
            scaler_a = StandardScaler().fit(train_box[features])
            m_audit = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42).fit(scaler_a.transform(train_box[features]), train_box['Log_Ret'])
            
            # Results
            a_preds = m_audit.predict(scaler_a.transform(test_box[features]))
            test_box['Strategy_Ret'] = np.sign(a_preds) * test_box['Log_Ret']
            test_box['Equity'] = (1 + test_box['Strategy_Ret']).cumprod() * 10000
            test_box['Market'] = (1 + test_box['Log_Ret']).cumprod() * 10000
            
            # --- NEW: MAX DRAWDOWN MATH ---
            peak = test_box['Equity'].expanding(min_periods=1).max()
            dd = (test_box['Equity'] - peak) / peak
            max_dd = dd.min() * 100
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Audit Accuracy", f"{(np.sign(a_preds) == np.sign(test_box['Log_Ret'])).mean()*100:.1f}%")
            c2.metric("Final Equity", f"${test_box['Equity'].iloc[-1]:,.2f}")
            c3.metric("Max Drawdown", f"{max_dd:.2f}%", delta_color="inverse")
            
            fig_audit = go.Figure()
            fig_audit.add_trace(go.Scatter(x=test_box.index, y=test_box['Equity'], name="AI Strategy", line=dict(color="#00ffcc")))
            fig_audit.add_trace(go.Scatter(x=test_box.index, y=test_box['Market'], name="Market", line=dict(color="gray", dash='dash')))
            st.plotly_chart(fig_audit, use_container_width=True)
            
            st.info(f"**Drawdown Analysis:** The scariest moment for this strategy was a **{max_dd:.2f}%** drop from its peak. This helps you understand the 'stomach' required to follow the AI's signals.")
