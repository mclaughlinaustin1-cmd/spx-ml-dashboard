import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

# Safety check for the Sentiment library
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

# --- Page Setup ---
st.set_page_config(layout="wide", page_title="AI Quant Terminal V3.2", page_icon="🏦")
st.markdown("<style>.main {background-color: #0b0e14;}</style>", unsafe_allow_html=True)

# --- Sentiment Analysis Engine ---
def get_sentiment(ticker):
    """Scrapes news and returns a sentiment score using VADER."""
    if not VADER_AVAILABLE:
        return 0, [{"title": "Sentiment Engine Offline - Update requirements.txt", "publisher": "System", "link": "#"}]
    
    try:
        tk = yf.Ticker(ticker)
        news = tk.news[:10]
        analyzer = SentimentIntensityAnalyzer()
        scores = [analyzer.polarity_scores(n['title'])['compound'] for n in news]
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
    
    # Handle yfinance MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Feature Engineering
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # RSI Calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    # Volatility and Bollinger Bands
    df['Vol_10'] = df['Log_Ret'].rolling(10).std()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['BB_Upper'] = df['MA20'] + (df['Close'].rolling(20).std() * 2)
    df['BB_Lower'] = df['MA20'] - (df['Close'].rolling(20).std() * 2)
    
    df = df.dropna()
    
    # ML Setup
    features = ['RSI', 'Vol_10', 'Log_Ret']
    X = df[features].values
    y = df['Log_Ret'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train on 5 years, Test on last 100 days
    split = len(df) - 100
    rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_scaled[:split], y[:split])
    
    # Win Rate (Directional Accuracy)
    test_preds = rf.predict(X_scaled[split:])
    win_rate = np.mean(np.sign(test_preds) == np.sign(y[split:])) * 100
    
    return df, rf, win_rate, scaler

# --- Main Dashboard ---
st.title("🏛️ Institutional AI Quant Terminal")

with st.sidebar:
    st.header("⚙️ Terminal Config")
    ticker_input = st.text_input("Enter Ticker", "NVDA").upper()
    target_date = st.date_input("Target Date", datetime.now() + timedelta(days=10))

result = build_quant_model(ticker_input)

if result:
    df, model, win_rate, scaler = result
    sentiment_score, news_list = get_sentiment(ticker_input)
    
    # --- Metrics Bar ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model Win Rate", f"{win_rate:.1f}%")
    mood = "Bullish" if sentiment_score > 0.05 else "Bearish" if sentiment_score < -0.05 else "Neutral"
    m2.metric("News Sentiment", mood, f"{sentiment_score:.2f}")
    m3.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
    m4.metric("Volatility (10d)", f"{df['Vol_10'].iloc[-1]*100:.2f}%")

    # --- Prediction ---
    days_out = (pd.Timestamp(target_date) - df.index[-1]).days
    last_state = scaler.transform(df[['RSI', 'Vol_10', 'Log_Ret']].tail(1).values)
    pred_ret = model.predict(last_state)[0]
    # Sentiment bias adjustment
    adj_ret = pred_ret + (sentiment_score * 0.002)
    final_price = df['Close'].iloc[-1] * np.exp(adj_ret * days_out)

    st.subheader(f"🔮 AI Target for {target_date}: ${final_price:,.2f}")

    # --- Visuals ---
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    
    # Candlestick
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='rgba(0, 255, 204, 0.2)'), name="Upper Band"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='rgba(0, 255, 204, 0.2)'), fill='tonexty', name="Lower Band"), row=1, col=1)
    
    # Prediction Path
    fig.add_trace(go.Scatter(x=[df.index[-1], pd.Timestamp(target_date)], y=[df['Close'].iloc[-1], final_price], line=dict(color='#ff9500', width=4, dash='dashdot'), name="AI Path"), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#ff9500'), name="RSI"), row=2, col=1)
    fig.update_layout(template="plotly_dark", height=800, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📰 Market News & Sentiment Analysis"):
        for n in news_list:
            st.markdown(f"**[{n['publisher']}]** {n['title']} ([Link]({n['link']}))")
else:
    st.error("Invalid Ticker or Connection Error.")
