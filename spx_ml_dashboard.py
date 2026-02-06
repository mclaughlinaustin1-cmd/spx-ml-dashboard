import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

# --- Robust VADER Import ---
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

# --- Core Model Logic ---
@st.cache_data(ttl=3600)
def build_quant_data(ticker):
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
    
    # ML Setup
    X = df[['RSI', 'Vol_10', 'Log_Ret']].values
    y = df['Log_Ret'].values
    scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X)
    
    split = len(df) - 100
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_s[:split], y[:split])
    
    # Win Rate
    preds = model.predict(X_s[split:])
    win_rate = (np.sign(preds) == np.sign(y[split:])).mean() * 100
    
    return df, model, win_rate, scaler

# --- Sentiment Engine ---
def get_sentiment(ticker):
    if not VADER_AVAILABLE: return 0
    try:
        tk = yf.Ticker(ticker)
        news = tk.news[:5]
        analyzer = SentimentIntensityAnalyzer()
        scores = [analyzer.polarity_scores(n['title'])['compound'] for n in news]
        return np.mean(scores) if scores else 0
    except: return 0

# --- UI Setup ---
st.set_page_config(layout="wide", page_title="AI Alpha Terminal")
st.title("🏛️ AI Alpha Terminal")

ticker = st.sidebar.text_input("Ticker", "NVDA").upper()
target_dt = st.sidebar.date_input("Target Date", datetime.now() + timedelta(days=7))

data = build_quant_data(ticker)

if data:
    df, model, win_rate, scaler = data
    sent_score = get_sentiment(ticker)
    
    # --- Prediction ---
    days_out = (pd.Timestamp(target_dt) - df.index[-1]).days
    current_state = scaler.transform(df[['RSI', 'Vol_10', 'Log_Ret']].tail(1).values)
    pred_ret = model.predict(current_state)[0]
    final_p = df['Close'].iloc[-1] * np.exp((pred_ret + (sent_score * 0.002)) * days_out)

    # --- Decision Matrix Logic ---
    reasons = []
    signals = {"BUY": 0, "SELL": 0}
    
    # RSI
    rsi_val = df['RSI'].iloc[-1]
    sig = "BUY" if rsi_val < 35 else "SELL" if rsi_val > 65 else "HOLD"
    reasons.append({"Factor": "RSI", "Status": f"Level {rsi_val:.1f}", "Signal": sig})
    if sig in signals: signals[sig] += 1

    # Bollinger
    price = df['Close'].iloc[-1]
    sig = "BUY" if price < df['BB_Low'].iloc[-1] else "SELL" if price > df['BB_Up'].iloc[-1] else "HOLD"
    reasons.append({"Factor": "BBands", "Status": "Price/Band Rel", "Signal": sig})
    if sig in signals: signals[sig] += 1

    # Final Decision Calculation
    decision = "STRONG BUY" if signals["BUY"] >= 2 else "BUY" if signals["BUY"] == 1 else "SELL" if signals["SELL"] >= 1 else "HOLD"
    color = "#00ffcc" if "BUY" in decision else "#ff4b4b" if "SELL" in decision else "#ff9500"

    # Header UI
    st.markdown(f"""<div style="border:2px solid {color}; padding:20px; border-radius:10px; text-align:center;">
        <h1 style="color:{color};">RECOMMENDATION: {decision}</h1>
        <p>Target Price: ${final_p:,.2f} | Win Rate: {win_rate:.1f}%</p>
    </div>""", unsafe_allow_html=True)

    # --- Charts ---
    col1, col2 = st.columns([2, 1])
    with col1:
        # FIXED: make_subplots brackets and row_heights correctly closed
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        
        # Candles
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
        
        # BBands
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Up'], line=dict(color='gray', width=1), name="Upper BB"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], line=dict(color='gray', width=1), fill='tonexty', name="Lower BB"), row=1, col=1)
        
        # RSI
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='orange'), name="RSI"), row=2, col=1)
        
        fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📊 Reason Matrix")
        st.table(pd.DataFrame(reasons))
        
        st.subheader("📈 Signal Strength")
        sig_fig = go.Figure(go.Pie(labels=['Buy', 'Sell', 'Hold'], values=[signals['BUY'], signals['SELL'], 2-(signals['BUY']+signals['SELL'])]))
        sig_fig.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(sig_fig, use_container_width=True)

else:
    st.error("Connection Error. Ensure Ticker is valid.")
