import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- Page Config ---
st.set_page_config(layout="wide", page_title="AI Institutional Terminal", page_icon="🏛️")

# --- Native Sentiment Engine (No TextBlob Required) ---
FINANCIAL_LEXICON = {
    "positive": ["growth", "beat", "buy", "surge", "soaring", "profit", "dividend", "bullish", "expansion", "upgrade"],
    "negative": ["drop", "miss", "sell", "plunge", "lawsuit", "debt", "bearish", "layoffs", "downgrade", "scandal"]
}

def get_native_sentiment(news_list):
    if not news_list: return 0, "Neutral"
    score = 0
    for n in news_list:
        text = n['title'].lower()
        for word in FINANCIAL_LEXICON["positive"]:
            if word in text: score += 1
        for word in FINANCIAL_LEXICON["negative"]:
            if word in text: score -= 1
    
    avg_score = score / len(news_list) if news_list else 0
    label = "Bullish" if avg_score > 0.1 else "Bearish" if avg_score < -0.1 else "Neutral"
    return avg_score, label

# --- Logic Modules ---

def load_comprehensive_data(ticker):
    tk = yf.Ticker(ticker)
    df = tk.history(period="1y")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df, tk.news

def apply_institutional_logic(df, whale_threshold):
    # Bollinger Bands
    df["MA20"] = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["BB_High"] = df["MA20"] + (std * 2)
    df["BB_Low"] = df["MA20"] - (std * 2)
    
    # Whale Tracker ($ value of volume)
    df["Whale_Val"] = df["Volume"] * df["Close"]
    df["Whale_Alert"] = df["Whale_Val"] > whale_threshold
    
    # Volatility (Annualized)
    df["Daily_Ret"] = df["Close"].pct_change()
    df["Vol"] = df["Daily_Ret"].rolling(20).std() * np.sqrt(252)
    
    return df

# --- Sidebar UI ---
with st.sidebar:
    st.title("🛡️ Risk Control")
    ticker_input = st.text_input("Asset Ticker", "NVDA").upper()
    forecast_horizon = st.slider("Predictive Trend (Days)", 1, 14, 7)
    whale_limit = st.number_input("Whale Threshold ($)", value=5000000, step=1000000)
    st.divider()
    st.subheader("Global Macro Factors")
    pol_announcement = st.checkbox("Major Political News Today?", value=False)
    market_shift = st.select_slider("Market Sentiment Shift", options=["Bearish", "Neutral", "Bullish"], value="Neutral")

# --- Main Execution ---
df_raw, news_data = load_comprehensive_data(ticker_input)

if not df_raw.empty:
    df = apply_institutional_logic(df_raw.copy(), whale_limit)
    sent_score, sent_label = get_native_sentiment(news_data)
    
    # Prediction Logic (Quadratic Trend)
    y_vals = df["Close"].values
    x_vals = np.arange(len(y_vals))
    poly_coeffs = np.polyfit(x_vals, y_vals, 2)
    future_x = np.arange(len(y_vals), len(y_vals) + forecast_horizon)
    forecast_y = np.polyval(poly_coeffs, future_x)
    future_dates = [df.index[-1] + timedelta(days=i+1) for i in range(forecast_horizon)]

    # --- UI Layout ---
    t1, t2, t3 = st.tabs(["📊 Market Analysis", "🧠 AI Sentiment", "⚠️ Risk Decision Tree"])

    with t1:
        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Live Price", f"${df['Close'].iloc[-1]:.2f}")
        m2.metric("Annual Vol", f"{df['Vol'].iloc[-1]*100:.1f}%")
        m3.metric("Sentiment", sent_label)
        m4.metric("Last Whale Tx", f"${df['Whale_Val'].iloc[-1]/1e6:.1f}M")

        # Visuals
        fig = go.Figure()
        # Price & Forecast
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Historical", line=dict(color='white', width=1.5)))
        fig.add_trace(go.Scatter(x=future_dates, y=forecast_y, name="Predictive Trend", line=dict(dash='dash', color='#00ffcc')))
        # Whale Highlighters
        whales = df[df["Whale_Alert"]]
        fig.add_trace(go.Scatter(x=whales.index, y=whales["Close"], mode="markers", name="Whale Spike", marker=dict(color="gold", size=8)))
        
        fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        st.subheader("Keyword-Based News Analysis")
        cols = st.columns(len(news_data[:6]))
        for idx, n in enumerate(news_data[:6]):
            with st.container():
                st.markdown(f"**{n['publisher']}**: {n['title']}")
                st.caption(f"[Read Article]({n['link']})")
                st.divider()

    with t3:
        st.subheader("Decision Logic Tree")
        
        # Risk Scoring Calculation
        risk_score = 0
        if df['Vol'].iloc[-1] > 0.4: risk_score += 30
        if sent_label == "Bearish": risk_score += 20
        if df['Whale_Alert'].iloc[-1]: risk_score += 20
        if pol_announcement: risk_score += 15
        if market_shift == "Bearish": risk_score += 15
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Aggregate Risk Score", f"{risk_score}/100")
            if risk_score > 60:
                st.error("RATING: HIGH RISK - AVOID")
            elif risk_score > 30:
                st.warning("RATING: MODERATE - CAUTION")
            else:
                st.success("RATING: LOW RISK - ACCUMULATE")

        with c2:
            st.markdown("### Risk Analysis Visualization")
            # Creating a decision tree visual representation
            st.code(f"""
            [START: {ticker_input} Analysis]
               |
               |-- Volatility > 40%? ----> {'[YES] +30pts' if df['Vol'].iloc[-1] > 0.4 else '[NO]'}
               |-- Sentiment Bearish? ---> {'[YES] +20pts' if sent_label == "Bearish" else '[NO]'}
               |-- Political Unrest? ----> {'[YES] +15pts' if pol_announcement else '[NO]'}
               |-- Whale Movement? ------> {'[YES] +20pts' if df['Whale_Alert'].iloc[-1] else '[NO]'}
               |
            [TOTAL SCORE: {risk_score}]
            """)
            
        

        # Final Running Trend with Volatility overlay
        st.subheader("Running Volatility & Risk Trend")
        st.line_chart(df['Vol'])

else:
    st.error("Ticker not found. Please check your spelling.")
