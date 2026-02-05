import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from textblob import TextBlob

# --- Page Config & UI ---
st.set_page_config(layout="wide", page_title="AI Institutional Terminal v2")

st.markdown("""
    <style>
    .reportview-container { background: #0e1117; }
    .metric-card { background: #161b22; border: 1px solid #30363d; padding: 20px; border-radius: 10px; }
    .risk-high { color: #ff4b4b; font-weight: bold; }
    .risk-low { color: #00ffcc; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- Logic Modules ---

def get_data_and_news(ticker, period):
    tk = yf.Ticker(ticker)
    df = tk.history(period=period)
    news = tk.news
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df, news

def analyze_sentiment(news_list):
    if not news_list: return 0, "Neutral"
    scores = [TextBlob(n['title']).sentiment.polarity for n in news_list]
    avg_score = np.mean(scores)
    sentiment = "Bullish" if avg_score > 0.05 else "Bearish" if avg_score < -0.05 else "Neutral"
    return avg_score, sentiment

def whale_tracker(df, threshold=5000000):
    # Estimate transaction value: Volume * Close Price
    df['Tx_Value'] = df['Volume'] * df['Close']
    df['Whale_Alert'] = df['Tx_Value'] > threshold
    return df

def run_risk_analysis(df, sentiment_score, vol_threshold):
    # Logic-based Decision Tree for Risk
    volatility = df['Close'].pct_change().std() * np.sqrt(252)
    latest_whale = df['Whale_Alert'].iloc[-1]
    
    risk_points = 0
    reasons = []
    
    if volatility > 0.30: 
        risk_points += 40
        reasons.append("High Market Volatility")
    if sentiment_score < 0: 
        risk_points += 20
        reasons.append("Negative News Sentiment")
    if latest_whale: 
        risk_points += 30
        reasons.append("Massive Whale Transaction Detected")
        
    risk_level = "High" if risk_points > 60 else "Moderate" if risk_points > 30 else "Low"
    return risk_level, risk_points, reasons

# --- Sidebar UI ---
with st.sidebar:
    st.title("🛡️ Risk Control")
    ticker = st.text_input("Ticker", "NVDA").upper()
    forecast_days = st.slider("Predictive Horizon (Days)", 1, 14, 7)
    whale_limit = st.number_input("Whale Threshold ($)", value=5000000, step=1000000)
    st.divider()
    st.info("Decision Tree includes: Volume Spikes, Sentiment Polarity, and Annualized Volatility.")

# --- Main Execution ---
df, news = get_data_and_news(ticker, "1y")

if not df.empty:
    df = whale_tracker(df, whale_limit)
    sent_score, sent_label = analyze_sentiment(news)
    risk_lv, risk_pts, risk_reasons = run_risk_analysis(df, sent_score, whale_limit)

    # Autoscale & Forecast Logic
    y = df["Close"].values
    x = np.arange(len(y))
    poly = np.polyfit(x, y, 2) # Quadratic for better curve
    future_x = np.arange(len(y), len(y) + forecast_days)
    forecast = np.polyval(poly, future_x)
    future_dates = [df.index[-1] + timedelta(days=i+1) for i in range(forecast_days)]

    # --- UI Layout ---
    t1, t2, t3 = st.tabs(["📊 Market Analysis", "🧠 AI Sentiment", "⚠️ Risk Decision Tree"])

    with t1:
        fig = go.Figure()
        # Price
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price", line=dict(color='white')))
        # Forecast
        fig.add_trace(go.Scatter(x=future_dates, y=forecast, name="AI Prediction", line=dict(dash='dot', color='cyan')))
        # Whale Markers
        whales = df[df['Whale_Alert']]
        fig.add_trace(go.Scatter(x=whales.index, y=whales['Close'], mode='markers', 
                                 name="Whale Tx > $5M", marker=dict(color='gold', size=10, symbol='diamond')))
        
        fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        st.subheader(f"Global Sentiment: {sent_label}")
        st.progress((sent_score + 1) / 2) # Normalize -1to1 to 0to1
        for n in news[:5]:
            st.write(f"📰 **{n['title']}**")
            st.caption(f"Source: {n['publisher']} | [Link]({n['link']})")

    with t3:
        col1, col2 = st.columns(2)
        with col1:
            st.header("Risk Analysis")
            st.metric("Risk Score", f"{risk_pts}/100", delta=risk_lv, delta_color="inverse")
            for r in risk_reasons:
                st.write(f"🚩 {r}")
        
        with col2:
            st.subheader("Automated Decision Path")
            # Visualizing the 'Logic Tree'
            st.code(f"""
IF Volatility > 30%  --> { "FAIL" if "High Market Volatility" in risk_reasons else "PASS" }
IF Sentiment > 0.0  --> { "PASS" if sent_score > 0 else "FAIL" }
IF Whale Spike < {whale_limit} --> { "PASS" if not df['Whale_Alert'].iloc[-1] else "FAIL" }
---------------------------
FINAL DECISION: { "HOLD/WATCH" if risk_lv == "High" else "EXECUTE TRADE" }
            """)
            
        # Risk Heatmap (Running Volatility)
        df['Vol'] = df['Close'].pct_change().rolling(20).std()
        st.line_chart(df['Vol'], use_container_width=True)

else:
    st.error("Invalid Ticker or No Data Found.")
