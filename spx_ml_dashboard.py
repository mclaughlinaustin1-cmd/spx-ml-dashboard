import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta

# --- Page Setup ---
st.set_page_config(layout="wide", page_title="AI Quant Terminal V3.1", page_icon="🏦")
st.markdown("<style>.main {background-color: #0b0e14;}</style>", unsafe_allow_html=True)

# --- Sentiment Analysis Engine ---
def get_sentiment(ticker):
    """Scrapes news and returns a sentiment score from -1 to 1."""
    try:
        tk = yf.Ticker(ticker)
        news = tk.news[:10]
        analyzer = SentimentIntensityAnalyzer()
        scores = []
        for n in news:
            scores.append(analyzer.polarity_scores(n['title'])['compound'])
        avg_score = np.mean(scores) if scores else 0
        return avg_score, news
    except:
        return 0, []

# --- ML & Backtesting Engine ---
@st.cache_data(ttl=3600)
def build_quant_model(ticker):
    # Fetch 5 years of data
    df = yf.download(ticker, period="5y", interval="1d")
    if df.empty: return None
    
    # Standardize column names (fixes yfinance MultiIndex issues)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Feature Engineering
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    # RSI Calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Volatility and Bollinger Bands
    df['Vol_10'] = df['Log_Ret'].rolling(10).std()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['BB_Upper'] = df['MA20'] + (df['Close'].rolling(20).std() * 2)
    df['BB_Lower'] = df['MA20'] - (df['Close'].rolling(20).std() * 2)
    
    df = df.dropna()
    
    # Features for the model
    features = ['RSI', 'Vol_10', 'Log_Ret']
    X = df[features].values
    
    # Prepare Model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Training (Walk-forward split)
    split = len(df) - 100
    rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_scaled[:split], df['Log_Ret'].iloc[:split])
    
    # Calculate Win Rate (Historical Prediction vs Reality)
    test_preds = rf.predict(X_scaled[split:])
    win_rate = (np.sign(test_preds) == np.sign(df['Log_Ret'].iloc[split:])).mean() * 100
    
    return df, rf, win_rate, scaler

# --- UI Execution ---
st.title("🏛️ Institutional AI Quant Terminal")

with st.sidebar:
    st.header("⚙️ Terminal Config")
    ticker = st.text_input("Enter Ticker", "NVDA").upper()
    target_date = st.date_input("Target Date", datetime.now() + timedelta(days=10))
    st.divider()
    st.info("Predicts directional movement using Log Returns and Random Forest Ensemble.")

# Fetch Data and Run Model
result = build_quant_model(ticker)

if result:
    df, model, win_rate, scaler = result
    sentiment_score, news_list = get_sentiment(ticker)
    
    # --- Metrics Dashboard ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model Win Rate", f"{win_rate:.1f}%")
    m2.metric("News Sentiment", "Bullish" if sentiment_score > 0.05 else "Bearish" if sentiment_score < -0.05 else "Neutral", f"{sentiment_score:.2f}")
    m3.metric("Last Close", f"${df['Close'].iloc[-1]:.2f}")
    m4.metric("RSI Level", f"{df['RSI'].iloc[-1]:.1f}")

    # --- Prediction Logic ---
    # Days from last data point to user date
    days_out = (pd.Timestamp(target_date) - df.index[-1]).days
    
    # Get current feature state for prediction
    features = ['RSI', 'Vol_10', 'Log_Ret']
    last_features = df[features].tail(1).values
    current_feat_scaled = scaler.transform(last_features)
    
    # Model predicts log return for a single step
    base_pred_ret = model.predict(current_feat_scaled)[0]
    
    # Adjust for sentiment (News Impact Weight)
    final_daily_ret = base_pred_ret + (sentiment_score * 0.002) 
    
    # Extrapolate to target date: Price * e^(rt)
    predicted_price = df['Close'].iloc[-1] * np.exp(final_daily_ret * days_out)

    st.subheader(f"🔮 AI Price Target for {target_date}: ${predicted_price:,.2f}")

    # --- Standard Candlestick Chart ---
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    
    # Main Plot: Candlesticks
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], 
        name="Price Action"
    ), row=1, col=1)
    
    # Bollinger Bands for context
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='rgba(0, 255, 204, 0.2)'), name="BB Upper"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='rgba(0, 255, 204, 0.2)'), fill='tonexty', name="BB Lower"), row=1, col=1)
    
    # Forecast Projection
    future_range = [df.index[-1], pd.Timestamp(target_date)]
    fig.add_trace(go.Scatter(
        x=future_range, y=[df['Close'].iloc[-1], predicted_price],
        line=dict(color='#ff9500', width=4, dash='dashdot'), name="AI Projected Path"
    ), row=1, col=1)
    
    # Indicator Plot: RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#ff9500'), name="RSI"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

    fig.update_layout(template="plotly_dark", height=850, xaxis_rangeslider_visible=False, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # --- News Sentiment Feed ---
    with st.expander("📰 Latest Headlines & Market Context"):
        for n in news_list:
            st.write(f"**{n['title']}**")
            st.caption(f"Source: {n['publisher']} | [View Article]({n['link']})")

else:
    st.error("Model Error: Ensure ticker symbol is correct and you have internet access.")
