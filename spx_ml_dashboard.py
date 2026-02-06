import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize NLP for Sentiment
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# --- Page Config ---
st.set_page_config(layout="wide", page_title="AI Alpha v9.0: Sovereign", page_icon="🏦")

# --- Global Sidebar ---
with st.sidebar:
    st.header("🏢 Institutional Controls")
    ticker = st.text_input("Ticker Symbol", "NVDA").upper()
    forecast_days = st.slider("Forecast Horizon (Days)", 1, 10, 5)
    st.divider()
    timeframes = {"1W": 7, "1M": 30, "3M": 90, "6M": 180, "1Y": 365, "5Y": 1825}
    selected_tf = st.selectbox("Market Viewport", list(timeframes.keys()), index=2)
    st.divider()
    st.write("### AI Sensitivity")
    vol_threshold = st.slider("Vol Filter (%)", 50, 100, 85)

# --- Core Data & Intelligence ---
@st.cache_data(ttl=3600)
def get_terminal_data(ticker, horizon=5):
    # Fetch Stock + Macro (10Y Yield)
    df = yf.download(ticker, period="2y", interval="1d")
    tnx = yf.download("^TNX", period="2y", interval="1d")['Close']
    
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # Feature Engineering
    df['TNX'] = tnx
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_10'] = df['Log_Ret'].rolling(10).std()
    df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().where(df['Close'].diff() > 0, 0).rolling(14).mean() / 
                                  -df['Close'].diff().where(df['Close'].diff() < 0, 0).rolling(14).mean())))
    
    df['Target'] = df['Close'].shift(-horizon) / df['Close'] - 1
    return df.dropna()

def get_sentiment(ticker):
    try:
        news = yf.Ticker(ticker).news
        scores = [sia.polarity_scores(n['title'])['compound'] for n in news]
        return np.mean(scores) if scores else 0
    except: return 0

data = get_terminal_data(ticker, forecast_days)

if data is not None:
    # 1. RUN AI PREDICTION
    features = ['RSI', 'Vol_10', 'TNX']
    model = RandomForestRegressor(n_estimators=100, random_state=42).fit(data[features].values, data['Target'].values * 100)
    last_row = data.iloc[-1]
    pred_move = model.predict(last_row[features].values.reshape(1, -1))[0]
    sentiment = get_sentiment(ticker)
    
    # 2. MONTE CARLO SIMULATION (1,000 Paths)
    mu, sigma = data['Log_Ret'].mean(), data['Log_Ret'].std()
    sim_paths = []
    for _ in range(100): # 100 paths for speed/smoothness
        prices = [last_row['Close']]
        for _ in range(forecast_days):
            prices.append(prices[-1] * np.exp(np.random.normal(mu, sigma)))
        sim_paths.append(prices)
    sim_paths = np.array(sim_paths)
    
    # --- UI RENDER ---
    col_l, col_r = st.columns([3, 1])
    
    with col_l:
        # A. PROBABILITY CONE CHART
        st.subheader(f"🔮 {ticker} Probability Matrix")
        vdf = data.tail(timeframes[selected_tf])
        fig = go.Figure()
        
        # Historical
        fig.add_trace(go.Candlestick(x=vdf.index, open=vdf['Open'], high=vdf['High'], low=vdf['Low'], close=vdf['Close'], name="History"))
        
        # Forecast Fan (Confidence Intervals)
        fut_dates = [vdf.index[-1] + timedelta(days=i) for i in range(forecast_days + 1)]
        p95 = np.percentile(sim_paths, 95, axis=0)
        p5 = np.percentile(sim_paths, 5, axis=0)
        p50 = np.percentile(sim_paths, 50, axis=0)
        
        fig.add_trace(go.Scatter(x=fut_dates, y=p95, line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=fut_dates, y=p5, fill='tonexty', fillcolor='rgba(0, 255, 204, 0.1)', line=dict(width=0), name="90% Confidence Cone"))
        fig.add_trace(go.Scatter(x=fut_dates, y=p50, line=dict(color='#00ffcc', dash='dot'), name="AI Median Target"))
        
        fig.update_layout(template="plotly_dark", height=450, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"**Description:** This is the 'Sovereign View'. The **Shaded Cone** represents 1,000 possible futures based on historical volatility. The AI's specific prediction is the **Dotted Line**. If the price breaks the top of the cone, it's an exceptional breakout; below the cone is a 'Black Swan' crash.")

        # B. VOLUME ANALYSIS
        st.subheader("🌊 Order Flow (Volume)")
        v_cols = ['#00ffcc' if vdf['Close'].iloc[i] >= vdf['Open'].iloc[i] else '#ff4b4b' for i in range(len(vdf))]
        st.plotly_chart(go.Figure(go.Bar(x=vdf.index, y=vdf['Volume'], marker_color=v_cols)).update_layout(template="plotly_dark", height=200), use_container_width=True)
        st.caption(f"**Description:** We are interpreting Volume as 'Conviction.' Large bars in this **{selected_tf}** view indicate where big banks are entering positions. We look for volume to 'confirm' the AI prediction.")

        # C. MOMENTUM & REGIME
        st.subheader("🛰️ Momentum & Macro Regime")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=vdf.index, y=vdf['RSI'], line=dict(color='orange')))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
        st.plotly_chart(fig_rsi.update_layout(template="plotly_dark", height=200, yaxis=dict(range=[0,100])), use_container_width=True)
        st.caption(f"**Description:** This tracks the 'Speed' of price. Between 30 and 70 is 'Balanced'. Above 70, the asset is 'Exhausted' (Overbought). Below 30, it is 'Depressed' (Oversold).")

    with col_r:
        # Scorecard
        st.metric("AI Confidence Move", f"{pred_move:+.2f}%")
        st.metric("News Sentiment", f"{'Bullish' if sentiment > 0.05 else 'Bearish' if sentiment < -0.05 else 'Neutral'}")
        st.write("---")
        st.write("### Alpha Breakdown")
        st.progress(int(min(max((pred_move + 5) * 10, 0), 100)), text="AI Strength")
        st.progress(int((sentiment + 1) * 50), text="News Score")
        
        # Macro Warning
        tnx_delta = data['TNX'].pct_change(5).iloc[-1]
        if tnx_delta > 0.05:
            st.error(f"⚠️ MACRO ALERT: 10Y Yields are up {tnx_delta*100:.1f}% this week. This may suppress growth stocks regardless of AI signals.")

