import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

# --- VADER Import Fallback ---
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

# --- Core Logic Functions ---
def get_sentiment_score(ticker):
    if not VADER_AVAILABLE: return 0
    try:
        tk = yf.Ticker(ticker)
        news = tk.news[:5]
        analyzer = SentimentIntensityAnalyzer()
        scores = [analyzer.polarity_scores(n['title'])['compound'] for n in news]
        return np.mean(scores) if scores else 0
    except: return 0

@st.cache_data(ttl=3600)
def build_quant_data(ticker):
    df = yf.download(ticker, period="5y", interval="1d")
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Technicals
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
    
    # ML Training
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

# --- Page UI ---
st.set_page_config(layout="wide", page_title="AI Alpha Terminal")
st.title("🏛️ AI Alpha Terminal: Decision Engine")

ticker = st.sidebar.text_input("Ticker", "AAPL").upper()
target_dt = st.sidebar.date_input("Target Date", datetime.now() + timedelta(days=7))

data = build_quant_data(ticker)

if data:
    df, model, win_rate, scaler = data
    sent_score = get_sentiment_score(ticker)
    
    # --- Prediction ---
    days_out = (pd.Timestamp(target_dt) - df.index[-1]).days
    last_state = scaler.transform(df[['RSI', 'Vol_10', 'Log_Ret']].tail(1).values)
    pred_ret = model.predict(last_state)[0]
    final_p = df['Close'].iloc[-1] * np.exp((pred_ret + (sent_score * 0.002)) * days_out)

    # --- DECISION LOGIC ---
    reasons = []
    buy_signals = 0
    sell_signals = 0

    # 1. RSI Logic
    rsi_val = df['RSI'].iloc[-1]
    if rsi_val < 35: 
        reasons.append({"Factor": "RSI", "Status": "Oversold", "Signal": "BUY", "Color": "#00ffcc"})
        buy_signals += 1
    elif rsi_val > 65: 
        reasons.append({"Factor": "RSI", "Status": "Overbought", "Signal": "SELL", "Color": "#ff4b4b"})
        sell_signals += 1
    else:
        reasons.append({"Factor": "RSI", "Status": "Neutral", "Signal": "HOLD", "Color": "#808080"})

    # 2. Bollinger Band Logic
    price = df['Close'].iloc[-1]
    if price < df['BB_Low'].iloc[-1]:
        reasons.append({"Factor": "Volatility", "Status": "Below Lower BB", "Signal": "BUY", "Color": "#00ffcc"})
        buy_signals += 1
    elif price > df['BB_Up'].iloc[-1]:
        reasons.append({"Factor": "Volatility", "Status": "Above Upper BB", "Signal": "SELL", "Color": "#ff4b4b"})
        sell_signals += 1
    else:
        reasons.append({"Factor": "Volatility", "Status": "Inside Bands", "Signal": "HOLD", "Color": "#808080"})

    # 3. Sentiment Logic
    if sent_score > 0.15:
        reasons.append({"Factor": "Sentiment", "Status": "Bullish News", "Signal": "BUY", "Color": "#00ffcc"})
        buy_signals += 1
    elif sent_score < -0.15:
        reasons.append({"Factor": "Sentiment", "Status": "Bearish News", "Signal": "SELL", "Color": "#ff4b4b"})
        sell_signals += 1
    else:
        reasons.append({"Factor": "Sentiment", "Status": "Neutral", "Signal": "HOLD", "Color": "#808080"})

    # 4. ML Prediction Logic
    proj_move = ((final_p / price) - 1) * 100
    if proj_move > 2.0:
        reasons.append({"Factor": "AI Model", "Status": f"Projected +{proj_move:.1f}%", "Signal": "BUY", "Color": "#00ffcc"})
        buy_signals += 1
    elif proj_move < -2.0:
        reasons.append({"Factor": "AI Model", "Status": f"Projected {proj_move:.1f}%", "Signal": "SELL", "Color": "#ff4b4b"})
        sell_signals += 1
    else:
        reasons.append({"Factor": "AI Model", "Status": "Flat Forecast", "Signal": "HOLD", "Color": "#808080"})

    # Final Decision Calculation
    if buy_signals >= 3: final_decision, decision_color = "STRONG BUY", "#00ffcc"
    elif buy_signals >= 2: final_decision, decision_color = "BUY", "#00ffcc"
    elif sell_signals >= 3: final_decision, decision_color = "STRONG SELL", "#ff4b4b"
    elif sell_signals >= 2: final_decision, decision_color = "SELL", "#ff4b4b"
    else: final_decision, decision_color = "HOLD", "#ff9500"

    # --- TOP ROW: FINAL DECISION ---
    st.markdown(f"""
        <div style="background-color:{decision_color}22; border: 2px solid {decision_color}; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 25px;">
            <h1 style="color:{decision_color}; margin:0;">FINAL RECOMMENDATION: {final_decision}</h1>
            <p style="color:white; margin:0;">Based on a combined analysis of Technicals, ML Forecasts, and Market Sentiment.</p>
        </div>
    """, unsafe_allow_html=True)

    # --- CHART & TABLE ---
    col1, col2 = st.columns([2, 1])

    with col1:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7
