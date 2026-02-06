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
st.set_page_config(layout="wide", page_title="AI Quant Terminal V3", page_icon="🏦")
st.markdown("<style>.main {background-color: #0b0e14;}</style>", unsafe_allow_html=True)

# --- Core Logic: Sentiment Analysis ---
def get_sentiment(ticker):
    """Scrapes news and returns a sentiment score from -1 to 1."""
    try:
        tk = yf.Ticker(ticker)
        news = tk.news[:10]  # Get latest 10 headlines
        analyzer = SentimentIntensityAnalyzer()
        scores = [analyzer.polarity_scores(n['title'])['compound'] for n in news]
        avg_score = np.mean(scores) if scores else 0
        return avg_score, news
    except:
        return 0, []

# --- Core Logic: ML & Backtesting ---
@st.cache_data(ttl=3600)
def build_quant_model(ticker):
    df = yf.download(ticker, period="5y", interval="1d")
    if df.empty: return None
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    
    # Feature Engineering
    df['Returns'] = df['Close'].pct_change()
    df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().where(df['Close'].diff() > 0, 0).rolling(14).mean() / 
                                  -df['Close'].diff().where(df['Close'].diff() < 0, 0).rolling(14).mean())))
    df['Vol_10'] = df['Returns'].rolling(10).std()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['BB_Upper'] = df['MA20'] + (df['Close'].rolling(20).std() * 2)
    df['BB_Lower'] = df['MA20'] - (df['Close'].rolling(20).std() * 2)
    
    df = df.dropna()
    
    # Backtest Logic (Last 100 days)
    features = ['RSI', 'Vol_10', 'Returns']
    X = df[features].values
    y = np.where(df['Close'].shift(-5) > df['Close'], 1, 0) # 1 if price is higher in 5 days
    
    split = len(df) - 100
    rf = RandomForestRegressor(n_estimators=100, max_depth=5)
    rf.fit(X[:split], df['Returns'].iloc[:split]) # Train on returns
    
    # Calculate Win Rate %
    test_preds = rf.predict(X[split:])
    win_rate = (np.sign(test_preds) == np.sign(df['Returns'].iloc[split:])).mean() * 100
    
    return df, rf, win_rate, scaler := StandardScaler().fit(X)

# --- UI Execution ---
st.title("🏛️ Institutional AI Quant Terminal")
ticker = st.sidebar.text_input("Enter Ticker", "NVDA").upper()
target_date = st.sidebar.date_input("Target Date", datetime.now() + timedelta(days=10))

data = build_quant_model(ticker)
if data:
    df, model, win_rate, scaler = data
    sentiment_score, news_list = get_sentiment(ticker)
    
    # --- Metrics Row ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model Win Rate (Hist)", f"{win_rate:.1f}%")
    sentiment_label = "Positive" if sentiment_score > 0.05 else "Negative" if sentiment_score < -0.05 else "Neutral"
    c2.metric("Market Sentiment", sentiment_label, f"{sentiment_score:.2f} Score")
    c3.metric("RSI (14d)", f"{df['RSI'].iloc[-1]:.1f}")
    c4.metric("Volatility", f"{df['Vol_10'].iloc[-1]*100:.2f}%")

    # --- Prediction Logic ---
    days_out = (pd.Timestamp(target_date) - df.index[-1]).days
    current_feat = scaler.transform(df[['RSI', 'Vol_10', 'Returns']].tail(1).values)
    pred_return = model.predict(current_feat)[0]
    # Adjust prediction based on sentiment
    adjusted_return = pred_return + (sentiment_score * 0.005) 
    final_price = df['Close'].iloc[-1] * (1 + adjusted_return * days_out)

    st.subheader(f"🔮 AI Target: ${final_price:,.2f} on {target_date}")

    # --- Advanced Candlestick Chart ---
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    
    # Candlesticks
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='rgba(173, 204, 255, 0.3)'), name="BB Upper"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='rgba(173, 204, 255, 0.3)'), fill='tonexty', name="BB Lower"), row=1, col=1)
    
    # Forecast Cone
    future_dates = [df.index[-1], pd.Timestamp(target_date)]
    fig.add_trace(go.Scatter(x=future_dates, y=[df['Close'].iloc[-1], final_price], line=dict(color='#00ffcc', width=4, dash='dashdot'), name="AI Path"), row=1, col=1)
    
    # RSI Subplot
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#ff9500'), name="RSI"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

    fig.update_layout(template="plotly_dark", height=800, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- News Sentiment Feed ---
    with st.expander("📰 Latest Institutional News & Sentiment"):
        for n in news_list:
            st.write(f"**{n['title']}**")
            st.caption(f"Source: {n['publisher']} | [Link]({n['link']})")
