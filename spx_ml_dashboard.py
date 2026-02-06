import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

# --- VADER Import with Diagnostic ---
VADER_AVAILABLE = False
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ModuleNotFoundError:
    st.error("🚨 **System Error: 'vaderSentiment' library not found.**")
    st.info("Check your GitHub for a file named **requirements.txt** containing the word: `vaderSentiment`")

# --- Page Config ---
st.set_page_config(layout="wide", page_title="AI Quant Terminal V3.3", page_icon="📈")

# --- Sentiment Logic ---
def get_sentiment(ticker):
    if not VADER_AVAILABLE:
        return 0, []
    try:
        tk = yf.Ticker(ticker)
        news = tk.news[:10]
        analyzer = SentimentIntensityAnalyzer()
        scores = [analyzer.polarity_scores(n['title'])['compound'] for n in news]
        return np.mean(scores) if scores else 0, news
    except:
        return 0, []

# --- Data & ML Engine ---
@st.cache_data(ttl=3600)
def build_model(ticker):
    df = yf.download(ticker, period="5y", interval="1d")
    if df.empty: return None
    
    # Fix Column Index
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Feature Engineering
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_10'] = df['Log_Ret'].rolling(10).std()
    
    # RSI Calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    # Bollinger Bands
    df['MA20'] = df['Close'].rolling(20).mean()
    df['BB_Up'] = df['MA20'] + (df['Close'].rolling(20).std() * 2)
    df['BB_Low'] = df['MA20'] - (df['Close'].rolling(20).std() * 2)
    
    df = df.dropna()
    
    # ML Setup
    X = df[['RSI', 'Vol_10', 'Log_Ret']].values
    y = df['Log_Ret'].values
    scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X)
    
    # Train / Backtest
    split = len(df) - 100
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_s[:split], y[:split])
    
    # Calculate Win Rate
    preds = model.predict(X_s[split:])
    win_rate = (np.sign(preds) == np.sign(y[split:])).mean() * 100
    
    return df, model, win_rate, scaler

# --- Execution ---
st.title("🏛️ Institutional AI Quant Terminal")
ticker = st.sidebar.text_input("Ticker", "AAPL").upper()
target_date = st.sidebar.date_input("Target Date", datetime.now() + timedelta(days=7))

data = build_model(ticker)

if data:
    df, model, win_rate, scaler = data
    sent_score, news = get_sentiment(ticker)
    
    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Win Rate", f"{win_rate:.1f}%")
    c2.metric("Sentiment", "Bullish" if sent_score > 0.05 else "Bearish" if sent_score < -0.05 else "Neutral", f"{sent_score:.2f}")
    c3.metric("Last Price", f"${df['Close'].iloc[-1]:.2f}")
    c4.metric("Volatility", f"{df['Vol_10'].iloc[-1]*100:.2f}%")

    # Predict Target Price
    days_out = (pd.Timestamp(target_date) - df.index[-1]).days
    current_state = scaler.transform(df[['RSI', 'Vol_10', 'Log_Ret']].tail(1).values)
    pred_ret = model.predict(current_state)[0] + (sent_score * 0.002)
    final_p = df['Close'].iloc[-1] * np.exp(pred_ret * days_out)

    st.subheader(f"🔮 Target for {target_date}: ${final_p:,.2f}")

    # Candlestick UI
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Up'], line=dict(color='gray', dash='dot'), name="BB Upper"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], line=dict(color='gray', dash='dot'), fill='tonexty', name="BB Lower"), row=1, col=1)
    
    # RSI Plot
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color="orange")), row=2, col=1)
    fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)


