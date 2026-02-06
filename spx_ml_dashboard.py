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

# --- Page Setup ---
st.set_page_config(layout="wide", page_title="AI Alpha Terminal v5.1")
st.title("🏛️ AI Alpha Terminal: Integrated Decision Engine")

@st.cache_data(ttl=3600)
def build_quant_data(ticker):
    df = yf.download(ticker, period="5y", interval="1d")
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # 1. Indicator Calculations
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
    
    # 2. Truth-Tested Accuracy Calculation
    X = df[['RSI', 'Vol_10', 'Log_Ret']].values
    y = df['Log_Ret'].values
    scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X)
    
    tscv = TimeSeriesSplit(n_splits=5)
    win_rates = []
    threshold = 0.005  # 0.5% Accuracy Threshold
    
    for train_idx, test_idx in tscv.split(X_s):
        fold_model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        fold_model.fit(X_s[train_idx], y[train_idx])
        preds = fold_model.predict(X_s[test_idx])
        actuals = y[test_idx]
        # Win = Right direction AND move was significant (>0.5%)
        wins = (np.sign(preds) == np.sign(actuals)) & (np.abs(actuals) >= threshold)
        win_rates.append(wins.mean())
    
    # 3. Final Production Model
    final_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    final_model.fit(X_s, y)
    
    return df, final_model, np.mean(win_rates)*100, scaler

# --- Main App Execution ---
ticker = st.sidebar.text_input("Ticker", "TSLA").upper()
target_dt = st.sidebar.date_input("Forecast Date", datetime.now() + timedelta(days=7))

data = build_quant_data(ticker)

if data:
    df, model, win_rate, scaler = data
    
    # AI Prediction Logic
    last_state = scaler.transform(df[['RSI', 'Vol_10', 'Log_Ret']].tail(1).values)
    pred_ret = model.predict(last_state)[0]
    days_out = (pd.Timestamp(target_dt) - df.index[-1]).days
    final_p = df['Close'].iloc[-1] * np.exp(pred_ret * days_out)
    proj_move = ((final_p / df['Close'].iloc[-1]) - 1) * 100

    # --- DECISION LOGIC ENGINE ---
    signals = {"BUY": 0, "SELL": 0, "HOLD": 0}
    reasons = []

    # Pillar 1: RSI
    rsi_val = df['RSI'].iloc[-1]
    if rsi_val < 30: signals["BUY"] += 1; reasons.append("RSI: Oversold")
    elif rsi_val > 70: signals["SELL"] += 1; reasons.append("RSI: Overbought")
    else: signals["HOLD"] += 1; reasons.append("RSI: Neutral")

    # Pillar 2: Bollinger Bands
    price = df['Close'].iloc[-1]
    if price < df['BB_Low'].iloc[-1]: signals["BUY"] += 1; reasons.append("Price: Below Lower BB")
    elif price > df['BB_Up'].iloc[-1]: signals["SELL"] += 1; reasons.append("Price: Above Upper BB")
    else: signals["HOLD"] += 1; reasons.append("Price: Inside Bands")

    # Pillar 3: AI Model Forecast (Threshold-Aware)
    if proj_move > 0.5: signals["BUY"] += 1; reasons.append(f"AI: Bullish (+{proj_move:.1f}%)")
    elif proj_move < -0.5: signals["SELL"] += 1; reasons.append(f"AI: Bearish ({proj_move:.1f}%)")
    else: signals["HOLD"] += 1; reasons.append("AI: Flat Forecast")

    # Final Decision Calculation
    if signals["BUY"] >= 2: final_decision, decision_color = "BUY", "#00ffcc"
    elif signals["SELL"] >= 2: final_decision, decision_color = "SELL", "#ff4b4b"
    else: final_decision, decision_color = "HOLD", "#ff9500"

    # --- Visual Display ---
    st.markdown(f"""
        <div style="background-color:{decision_color}22; border: 2px solid {decision_color}; padding: 20px; border-radius: 10px; text-align: center;">
            <h1 style="color:{decision_color}; margin:0;">SIGNAL: {final_decision}</h1>
            <p style="color:white; margin:0;"><b>Reasoning:</b> {', '.join(reasons)}</p>
        </div>
    """, unsafe_allow_html=True)

    # --- Charts & Methodology ---
    col1, col2 = st.columns([2, 1])
    with col1:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Up'], line=dict(color='rgba(255,255,255,0.2)'), name="Upper BB"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], line=dict(color='rgba(255,255,255,0.2)'), fill='tonexty', name="Lower BB"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='orange'), name="RSI"), row=2, col=1)
        fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.metric("Truth-Tested Win Rate", f"{win_rate:.1f}%")
        st.metric("Predicted Target", f"${final_p:,.2f}")
        st.info("**Methodology:** The win rate counts correct directional guesses where the actual move was >0.5%. Signals require 2/3 indicators to agree.")

    # Methodology Expander
    with st.expander("📖 View Data Definitions"):
        st.latex(r"Win\ Rate = \frac{\text{Correct Predictions} \cap |\text{Actual Return}| > 0.5\%}{\text{Total Predictions}}")
        st.write("This prevents '100% Accuracy' bugs by ignoring tiny, untradable market noise.")

else:
    st.error("Invalid Ticker or Data Connection Lost.")
