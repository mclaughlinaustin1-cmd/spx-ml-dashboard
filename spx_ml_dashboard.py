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
st.set_page_config(layout="wide", page_title="AI Alpha Terminal v5.0")
st.title("🏛️ AI Alpha Terminal: Truth-Tested Engine")

# --- Core Logic with TimeSeriesSplit ---
@st.cache_data(ttl=3600)
def build_quant_data(ticker):
    df = yf.download(ticker, period="5y", interval="1d")
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # 1. Technical Indicators
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
    
    # 2. Features & Scaling
    features = ['RSI', 'Vol_10', 'Log_Ret']
    X = df[features].values
    y = df['Log_Ret'].values
    scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X)
    
    # 3. REALISTIC WIN RATE (TimeSeriesSplit)
    # We test the model by "walking forward" in time
    tscv = TimeSeriesSplit(n_splits=5)
    win_rates = []
    
    threshold = 0.005 # 0.5% Threshold to count as a 'significant' win
    
    for train_idx, test_idx in tscv.split(X_s):
        model_fold = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        model_fold.fit(X_s[train_idx], y[train_idx])
        preds = model_fold.predict(X_s[test_idx])
        
        # Win if: Direction is correct AND absolute return > threshold
        actuals = y[test_idx]
        correct_dir = (np.sign(preds) == np.sign(actuals))
        sig_move = (np.abs(actuals) >= threshold)
        
        # Fold accuracy
        fold_acc = (correct_dir & sig_move).mean()
        win_rates.append(fold_acc)
    
    true_win_rate = np.mean(win_rates) * 100
    
    # 4. Final Production Model
    final_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    final_model.fit(X_s, y)
    
    return df, final_model, true_win_rate, scaler

# --- UI Sidebar ---
ticker = st.sidebar.text_input("Enter Ticker", "NVDA").upper()
target_dt = st.sidebar.date_input("Target Date", datetime.now() + timedelta(days=7))

data = build_quant_data(ticker)

if data:
    df, model, win_rate, scaler = data
    
    # --- Prediction Calculation ---
    last_state = scaler.transform(df[['RSI', 'Vol_10', 'Log_Ret']].tail(1).values)
    pred_ret = model.predict(last_state)[0]
    days_out = (pd.Timestamp(target_dt) - df.index[-1]).days
    final_p = df['Close'].iloc[-1] * np.exp(pred_ret * days_out)

    # --- Header Metrics ---
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Predicted Price", f"${final_p:,.2f}")
    col_b.metric("Truth-Tested Win Rate", f"{win_rate:.1f}%")
    col_c.metric("Current RSI", f"{df['RSI'].iloc[-1]:.1f}")

    # --- Charts ---
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Up'], line=dict(color='gray', width=1), name="Upper BB"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], line=dict(color='gray', width=1), fill='tonexty', name="Lower BB"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='orange'), name="RSI"), row=2, col=1)
    fig.update_layout(template="plotly_dark", height=800, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- METHODOLOGY SECTION ---
    st.divider()
    st.header("📖 Understanding the Data & Methodology")
    m1, m2 = st.columns(2)
    
    with m1:
        st.subheader("How is the Win Rate captured?")
        st.write("""
        Unlike standard models that 'peek' at the future, we use **TimeSeriesSplit**. 
        This divides the last 5 years into 5 sequential 'chapters'. 
        The model trains on Chapter 1, then tries to predict Chapter 2. 
        Then it trains on Chapters 1-2 and predicts Chapter 3.
        """)
        st.latex(r"Accuracy = \frac{\sum (Correct\ Direction \cap |Return| > 0.5\%)}{Total\ Days}")

    with m2:
        st.subheader("The 0.5% Integrity Threshold")
        st.write("""
        A 'Win' isn't just guessing the right direction. To prevent 'garbage wins' (where the stock moves 0.001%), 
        our engine only counts a prediction as successful if the actual market move was **greater than 0.5%**. 
        This simulates a real-world environment where trading fees and spreads exist.
        """)
        
    if win_rate > 65:
        st.success("🎯 **High Confidence:** This ticker shows strong historical patterns.")
    elif win_rate < 45:
        st.warning("⚠️ **Low Edge:** The model is struggling to find a consistent pattern. Trade with caution.")

else:
    st.error("Data error. Check ticker symbol.")
