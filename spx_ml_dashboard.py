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
    st.warning("Sentiment engine is initializing. Please ensure 'vaderSentiment' is in requirements.txt.")

# --- Page Config ---
st.set_page_config(layout="wide", page_title="AI Quant Terminal v4.1", page_icon="🏦")
st.markdown("""
    <style>
    .report-card {
        background-color: #161b22;
        border-radius: 10px;
        padding: 20px;
        border-left: 5px solid #00ffcc;
        margin-top: 20px;
    }
    .stMetric { background-color: #0d1117; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

# --- Core Logic Functions ---
def get_sentiment_data(ticker):
    if not VADER_AVAILABLE: return 0, []
    try:
        tk = yf.Ticker(ticker)
        news = tk.news[:8]
        analyzer = SentimentIntensityAnalyzer()
        scores = [analyzer.polarity_scores(n['title'])['compound'] for n in news]
        return np.mean(scores) if scores else 0, news
    except: return 0, []

@st.cache_data(ttl=3600)
def build_advanced_model(ticker):
    # 1. Data Fetching
    df = yf.download(ticker, period="5y", interval="1d")
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # 2. Technical Feature Engineering
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
    
    # 3. Earnings Data (BUG FIX APPLIED HERE)
    tk = yf.Ticker(ticker)
    next_earnings = None
    try:
        calendar = tk.calendar
        if isinstance(calendar, pd.DataFrame) and not calendar.empty:
            next_earnings = calendar.iloc[0, 0]
        elif isinstance(calendar, dict) and 'Earnings Date' in calendar:
            next_earnings = calendar['Earnings Date'][0]
    except:
        pass

    # 4. Machine Learning
    df = df.dropna()
    X = df[['RSI', 'Vol_10', 'Log_Ret']].values
    y = df['Log_Ret'].values
    
    scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X)
    
    # Train / Backtest Split
    split = len(df) - 100
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_s[:split], y[:split])
    
    # Win Rate (Directional Accuracy)
    preds = model.predict(X_s[split:])
    win_rate = (np.sign(preds) == np.sign(y[split:])).mean() * 100
    
    return df, model, win_rate, scaler, next_earnings

# --- UI Layout ---
st.title("🏛️ Institutional AI Quant Terminal v4.1")
ticker_input = st.sidebar.text_input("Ticker Symbol", "TSLA").upper()
target_date = st.sidebar.date_input("Analysis Target Date", datetime.now() + timedelta(days=14))

data = build_advanced_model(ticker_input)

if data:
    df, model, win_rate, scaler, earnings_date = data
    sent_score, news = get_sentiment_data(ticker_input)
    
    # --- Metrics ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model Win-Rate", f"{win_rate:.1f}%")
    c2.metric("Market Sentiment", "Bullish" if sent_score > 0.05 else "Bearish", f"{sent_score:.2f}")
    c3.metric("RSI (14d)", f"{df['RSI'].iloc[-1]:.1f}")
    c4.metric("Volatility", f"{df['Vol_10'].iloc[-1]*100:.2f}%")

    # --- Prediction Calculation ---
    days_out = (pd.Timestamp(target_date) - df.index[-1]).days
    current_state = scaler.transform(df[['RSI', 'Vol_10', 'Log_Ret']].tail(1).values)
    raw_pred_ret = model.predict(current_state)[0]
    
    # Incorporate Sentiment Bias
    adj_ret = raw_pred_ret + (sent_score * 0.002)
    target_price = df['Close'].iloc[-1] * np.exp(adj_ret * days_out)

    # --- Main Candlestick Graph ---
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[df.index[-1], pd.Timestamp(target_date)], y=[df['Close'].iloc[-1], target_price], 
                             line=dict(color='#00ffcc', width=4, dash='dashdot'), name="AI Path"), row=1, col=1)
    fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- Intelligence Brief ---
    st.markdown(f"""
    <div class="report-card">
        <h3>🧠 Intelligence Brief: {ticker_input}</h3>
        <p>The <b>Random Forest Ensemble</b> (Win-Rate: {win_rate:.1f}%) predicts a target price of <b>${target_price:,.2f}</b> 
        for {target_date.strftime('%B %d, %Y')}. This represents a <b>{((target_price/df['Close'].iloc[-1])-1)*100:+.2f}%</b> move.</p>
        <p><b>Technical Justification:</b> The model identifies current RSI momentum and historical volatility patterns as the primary drivers. 
        Sentiment data from 8 news catalysts indicates a <b>{sent_score:.2f}</b> weighting on the expected return.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Catalyst & Risk Matrix ---
    st.divider()
    l_col, r_col = st.columns(2)
    
    with l_col:
        st.subheader("⚠️ Factor Risk Matrix")
        # Map risk factors (0.0 to 1.0 scale)
        vol_risk = min(1.0, df['Vol_10'].iloc[-1] * 60)
        rsi_risk = abs(df['RSI'].iloc[-1] - 50) / 50
        sent_risk = 1 - abs(sent_score)
        
        # Calculate Earnings Risk
        days_to_earn = 99
        if earnings_date:
            try:
                days_to_earn = (pd.Timestamp(earnings_date) - pd.Timestamp(target_date)).days
            except: pass
        earn_risk = 0.9 if 0 <= days_to_earn <= 7 else 0.2

        categories = ['Volatility', 'Earnings Risk', 'Sentiment Stability', 'RSI Overextension', 'Macro Drift']
        values = [vol_risk, earn_risk, sent_risk, rsi_risk, 0.4]

        fig_risk = go.Figure(data=go.Scatterpolar(r=values, theta=categories, fill='toself', line=dict(color='#00ffcc')))
        fig_risk.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), template="plotly_dark", height=400)
        st.plotly_chart(fig_risk, use_container_width=True)
        

    with r_col:
        st.subheader("📰 Recent Market Catalysts")
        for n in news[:4]:
            st.markdown(f"**{n['title']}**")
            st.caption(f"Source: {n['publisher']} | [Link]({n['link']})")
        
        if earnings_date:
            st.warning(f"🔔 **Upcoming Earnings Event:** {earnings_date}")
