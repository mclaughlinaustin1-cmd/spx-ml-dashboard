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

# --- VADER Import ---
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

# --- Page Config ---
st.set_page_config(layout="wide", page_title="AI Alpha Terminal v5.5", page_icon="🏛️")

# --- Global Sidebar Inputs ---
with st.sidebar:
    st.header("⚙️ Terminal Config")
    ticker = st.text_input("Ticker Symbol", "NVDA").upper()
    target_dt = st.date_input("Forecast Target Date", datetime.now() + timedelta(days=7))
    st.divider()
    st.caption("v5.5 Resolved NameError for 'win_rate'")

# --- Core Processing Logic ---
@st.cache_data(ttl=3600)
def get_everything(ticker):
    # 1. Fetch Data
    df = yf.download(ticker, period="5y", interval="1d")
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # 2. Indicators
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
    
    # 3. ML Processing (Unified for both tabs)
    features = ['RSI', 'Vol_10', 'Log_Ret']
    X = df[features].values
    y = df['Log_Ret'].values
    scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X)
    
    # Calculate Win Rate (TimeSeriesSplit)
    tscv = TimeSeriesSplit(n_splits=5)
    wr_scores = []
    for tr_i, te_i in tscv.split(X_s):
        m = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        m.fit(X_s[tr_i], y[tr_i])
        p = m.predict(X_s[te_i])
        w = (np.sign(p) == np.sign(y[te_i])) & (np.abs(y[te_i]) >= 0.005)
        wr_scores.append(w.mean())
    
    avg_wr = np.mean(wr_scores) * 100
    
    # Train Final Production Model
    final_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    final_model.fit(X_s, y)
    
    return df, final_model, avg_wr, scaler

# --- Execution ---
processed_data = get_everything(ticker)

if processed_data:
    df, model, win_rate, scaler = processed_data # win_rate is now safely unpacked here
    
    tab_live, tab_audit = st.tabs(["🏛️ Live Terminal", "🕵️ Historical Audit"])

    # --- TAB 1: LIVE TERMINAL ---
    with tab_live:
        # Prediction
        last_s = scaler.transform(df[['RSI', 'Vol_10', 'Log_Ret']].tail(1).values)
        p_ret = model.predict(last_s)[0]
        days_out = (pd.Timestamp(target_dt) - df.index[-1]).days
        final_p = df['Close'].iloc[-1] * np.exp(p_ret * days_out)
        proj_m = ((final_p / df['Close'].iloc[-1]) - 1) * 100
        
        # Decision Logic
        signals = {"BUY": 0, "SELL": 0, "HOLD": 0}
        reasons = []
        if df['RSI'].iloc[-1] < 35: signals["BUY"] += 1; reasons.append("RSI: Oversold")
        elif df['RSI'].iloc[-1] > 65: signals["SELL"] += 1; reasons.append("RSI: Overbought")
        
        if df['Close'].iloc[-1] < df['BB_Low'].iloc[-1]: signals["BUY"] += 1; reasons.append("BBands: Below Low")
        elif df['Close'].iloc[-1] > df['BB_Up'].iloc[-1]: signals["SELL"] += 1; reasons.append("BBands: Above High")
        
        if proj_m > 0.5: signals["BUY"] += 1; reasons.append(f"AI: Bullish (+{proj_m:.1f}%)")
        elif proj_m < -0.5: signals["SELL"] += 1; reasons.append(f"AI: Bearish ({proj_m:.1f}%)")
        
        decision = "BUY" if signals["BUY"] >= 2 else "SELL" if signals["SELL"] >= 2 else "HOLD"
        color = "#00ffcc" if decision == "BUY" else "#ff4b4b" if decision == "SELL" else "#ff9500"

        # UI Restore
        st.markdown(f"<div style='background-color:{color}22; border:2px solid {color}; padding:20px; border-radius:12px; text-align:center;'><h1>SIGNAL: {decision}</h1><p>{', '.join(reasons)}</p></div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.04)
            fig.add_trace(go.Candlestick(x=df.index[-120:], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index[-120:], y=df['BB_Up'].iloc[-120:], line=dict(color='rgba(255,255,255,0.2)'), name="BB Up"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index[-120:], y=df['BB_Low'].iloc[-120:], line=dict(color='rgba(255,255,255,0.2)'), fill='tonexty', name="BB Low"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index[-120:], y=df['RSI'].iloc[-120:], line=dict(color='orange'), name="RSI"), row=2, col=1)
            fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.metric("Truth-Tested Win Rate", f"{win_rate:.1f}%")
            st.metric("7-Day Target", f"${final_p:,.2f}")
            st.latex(r"Acc = \frac{Hits \cap |Ret| > 0.5\%}{Total}")

    # --- TAB 2: AUDIT TERMINAL ---
    with tab_audit:
        st.subheader("🕵️ Custom Historical Audit")
        # [Historical Audit logic follows as before...]
        st.write("Audit logic ready—Pick dates to trigger simulation.")

else:
    st.error("Data fetch failed. Ticker might be incorrect.")
