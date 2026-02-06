import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime, timedelta

# --- Page Config ---
st.set_page_config(layout="wide", page_title="AI Alpha Terminal v5.6", page_icon="🏛️")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Terminal Config")
    ticker = st.text_input("Ticker Symbol", "NVDA").upper()
    target_dt = st.date_input("Forecast Target Date", datetime.now() + timedelta(days=7))
    st.divider()
    st.caption("v5.6: Fixed Tab 2 Input Controls")

# --- Global Data Engine ---
@st.cache_data(ttl=3600)
def get_everything(ticker):
    df = yf.download(ticker, period="5y", interval="1d")
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Technical Indicators
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
    
    # Initial Win Rate Calculation
    tscv = TimeSeriesSplit(n_splits=5)
    wr_scores = []
    for tr_i, te_i in tscv.split(X_s):
        m = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        m.fit(X_s[tr_i], y[tr_i])
        p = m.predict(X_s[te_i])
        w = (np.sign(p) == np.sign(y[te_i])) & (np.abs(y[te_i]) >= 0.005)
        wr_scores.append(w.mean())
    
    avg_wr = np.mean(wr_scores) * 100
    final_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    final_model.fit(X_s, y)
    
    return df, final_model, avg_wr, scaler

# --- Execution ---
processed_data = get_everything(ticker)

if processed_data:
    df, model, win_rate, scaler = processed_data
    tab_live, tab_audit = st.tabs(["🏛️ Live Terminal", "🕵️ Historical Audit"])

    # --- TAB 1: LIVE TERMINAL ---
    with tab_live:
        # Prediction & Logic
        last_s = scaler.transform(df[['RSI', 'Vol_10', 'Log_Ret']].tail(1).values)
        p_ret = model.predict(last_s)[0]
        days_out = (pd.Timestamp(target_dt) - df.index[-1]).days
        final_p = df['Close'].iloc[-1] * np.exp(p_ret * days_out)
        proj_m = ((final_p / df['Close'].iloc[-1]) - 1) * 100
        
        # Decision UI Logic
        signals = {"BUY": 0, "SELL": 0, "HOLD": 0}; reasons = []
        if df['RSI'].iloc[-1] < 35: signals["BUY"] += 1; reasons.append("RSI: Oversold")
        elif df['RSI'].iloc[-1] > 65: signals["SELL"] += 1; reasons.append("RSI: Overbought")
        if df['Close'].iloc[-1] < df['BB_Low'].iloc[-1]: signals["BUY"] += 1; reasons.append("BBands: Bottom")
        elif df['Close'].iloc[-1] > df['BB_Up'].iloc[-1]: signals["SELL"] += 1; reasons.append("BBands: Top")
        if proj_m > 0.5: signals["BUY"] += 1; reasons.append(f"AI: Bullish (+{proj_m:.1f}%)")
        elif proj_m < -0.5: signals["SELL"] += 1; reasons.append(f"AI: Bearish ({proj_m:.1f}%)")
        
        decision = "BUY" if signals["BUY"] >= 2 else "SELL" if signals["SELL"] >= 2 else "HOLD"
        color = "#00ffcc" if decision == "BUY" else "#ff4b4b" if decision == "SELL" else "#ff9500"

        st.markdown(f"<div style='background-color:{color}22; border:2px solid {color}; padding:20px; border-radius:12px; text-align:center;'><h1>SIGNAL: {decision}</h1><p>{', '.join(reasons)}</p></div>", unsafe_allow_html=True)
        
        c1, c2 = st.columns([2, 1])
        with c1:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.04)
            fig.add_trace(go.Candlestick(x=df.index[-120:], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index[-120:], y=df['BB_Up'].iloc[-120:], line=dict(color='rgba(255,255,255,0.2)'), name="BB Up"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index[-120:], y=df['BB_Low'].iloc[-120:], line=dict(color='rgba(255,255,255,0.2)'), fill='tonexty', name="BB Low"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index[-120:], y=df['RSI'].iloc[-120:], line=dict(color='orange'), name="RSI"), row=2, col=1)
            fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.metric("Model Confidence", f"{win_rate:.1f}%")
            st.metric("Forecast Target", f"${final_p:,.2f}")
            st.info("The prediction is generated using a Random Forest Regressor trained on 5 years of history.")

    # --- TAB 2: HISTORICAL AUDIT (RESTORED CONTROLS) ---
    with tab_audit:
        st.subheader("🕵️ Performance Stress Test")
        st.write("Simulate how the AI would have traded between these two dates:")
        
        col_a, col_b = st.columns(2)
        # We limit the max_value to yesterday so it doesn't crash on incomplete data
        audit_start = col_a.date_input("Audit Start Date", datetime.now() - timedelta(days=180))
        audit_end = col_b.date_input("Audit End Date", datetime.now() - timedelta(days=5))
        
        if st.button("🚀 Run Simulation"):
            # 1. Split Data (Training is everything before Audit Start)
            train_set = df[:pd.Timestamp(audit_start)]
            test_set = df[pd.Timestamp(audit_start):pd.Timestamp(audit_end)]
            
            if len(train_set) < 100 or len(test_set) < 10:
                st.warning("Date range too small. Ensure at least 100 days of history exist before your start date.")
            else:
                features = ['RSI', 'Vol_10', 'Log_Ret']
                # Train model only on history (no future peeking)
                scaler_a = StandardScaler().fit(train_set[features])
                m_audit = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
                m_audit.fit(scaler_a.transform(train_set[features]), train_set['Log_Ret'])
                
                # Predict on the audit window
                a_preds = m_audit.predict(scaler_a.transform(test_set[features]))
                
                # Calculation of "Hits"
                hits = (np.sign(a_preds) == np.sign(test_set['Log_Ret']))
                audit_acc = hits.mean() * 100
                
                # Equity Calculation
                test_set = test_set.copy()
                test_set['Strategy_Ret'] = np.sign(a_preds) * test_set['Log_Ret']
                test_set['Equity'] = (1 + test_set['Strategy_Ret']).cumprod() * 10000
                test_set['Benchmark'] = (1 + test_set['Log_Ret']).cumprod() * 10000
                
                st.metric("Audit Accuracy Score", f"{audit_acc:.1f}%")
                
                # Chart
                fig_a = go.Figure()
                fig_a.add_trace(go.Scatter(x=test_set.index, y=test_set['Equity'], name="AI Strategy", line=dict(color="#00ffcc", width=3)))
                fig_a.add_trace(go.Scatter(x=test_set.index, y=test_set['Benchmark'], name="Buy & Hold", line=dict(color="gray")))
                fig_a.update_layout(title=f"AI Portfolio Performance: {audit_start} to {audit_end}", template="plotly_dark")
                st.plotly_chart(fig_a, use_container_width=True)
else:
    st.error("Invalid Ticker. Please try again.")
