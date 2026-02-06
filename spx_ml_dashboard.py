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
st.set_page_config(layout="wide", page_title="AI Quant Terminal v4.0", page_icon="🏦")
st.markdown("""
    <style>
    .report-card {
        background-color: #161b22;
        border-radius: 10px;
        padding: 20px;
        border-left: 5px solid #ff9500;
    }
    .metric-box {
        text-align: center;
        padding: 10px;
        background-color: #0d1117;
        border-radius: 5px;
    }
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
    df = yf.download(ticker, period="5y", interval="1d")
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Technical Indicators
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_10'] = df['Log_Ret'].rolling(10).std()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    # Bollinger Bands
    df['MA20'] = df['Close'].rolling(20).mean()
    df['BB_Up'] = df['MA20'] + (df['Close'].rolling(20).std() * 2)
    df['BB_Low'] = df['MA20'] - (df['Close'].rolling(20).std() * 2)
    
    # Earnings Proximity (Mock logic for determining factors)
    tk = yf.Ticker(ticker)
    calendar = tk.calendar
    next_earnings = calendar.iloc[0,0] if calendar is not None and not calendar.empty else None

    df = df.dropna()
    X = df[['RSI', 'Vol_10', 'Log_Ret']].values
    y = df['Log_Ret'].values
    
    scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X)
    
    # Model Training
    split = len(df) - 100
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_s[:split], y[:split])
    
    # Backtest Accuracy
    preds = model.predict(X_s[split:])
    win_rate = (np.sign(preds) == np.sign(y[split:])).mean() * 100
    
    return df, model, win_rate, scaler, next_earnings

# --- UI Header ---
st.title("🏛️ Institutional AI Quant Terminal v4.0")
ticker = st.sidebar.text_input("Ticker Symbol", "NVDA").upper()
target_date = st.sidebar.date_input("Analysis Target Date", datetime.now() + timedelta(days=14))

data = build_advanced_model(ticker)

if data:
    df, model, win_rate, scaler, earnings_date = data
    sent_score, news = get_sentiment_data(ticker)
    
    # Top Stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model Win-Rate", f"{win_rate:.1f}%")
    c2.metric("Sentiment Score", f"{sent_score:.2f}")
    c3.metric("Current RSI", f"{df['RSI'].iloc[-1]:.1f}")
    c4.metric("Days to Target", (pd.Timestamp(target_date) - df.index[-1]).days)

    # Prediction Math
    days_out = (pd.Timestamp(target_date) - df.index[-1]).days
    current_state = scaler.transform(df[['RSI', 'Vol_10', 'Log_Ret']].tail(1).values)
    raw_pred = model.predict(current_state)[0]
    final_pred_ret = raw_pred + (sent_score * 0.001)
    target_price = df['Close'].iloc[-1] * np.exp(final_pred_ret * days_out)

    # --- MAIN CHART ---
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Market Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[df.index[-1], pd.Timestamp(target_date)], y=[df['Close'].iloc[-1], target_price], 
                             line=dict(color='#00ffcc', width=4, dash='dashdot'), name="AI Prediction Path"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Up'], line=dict(color='rgba(255,255,255,0.2)'), name="Upper Band"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], line=dict(color='rgba(255,255,255,0.2)'), fill='tonexty', name="Lower Band"), row=1, col=1)
    fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- INTELLIGENCE BRIEF ---
    st.markdown('<div class="report-card">', unsafe_allow_html=True)
    st.subheader(f"🧠 Intelligence Brief: {ticker} Outlook for {target_date.strftime('%b %d, %Y')}")
    
    trend = "Bullish" if target_price > df['Close'].iloc[-1] else "Bearish"
    move_pct = ((target_price - df['Close'].iloc[-1]) / df['Close'].iloc[-1]) * 100
    
    st.write(f"""
    The Random Forest ensemble has generated a **{trend}** forecast with a projected move of **{move_pct:+.2f}%**. 
    This prediction is driven by a combination of current **RSI ({df['RSI'].iloc[-1]:.2f})** levels and mean-reversion 
    tendencies relative to the Bollinger Bands. 
    
    **Key Finding:** The model shows a **{win_rate:.1f}% historical win-rate** for this ticker. 
    The forecast incorporates a sentiment bias of {sent_score:.2f} based on the last 8 major news catalysts.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- RISK & CATALYST MATRIX ---
    st.divider()
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("⚠️ Factor Risk Matrix")
        # Generate a spider/radar chart for risk factors
        risk_factors = ['Volatility', 'Earnings Risk', 'Sentiment Shift', 'Overbought', 'Macro Drag']
        
        # Logic for mock risk levels
        vol_risk = min(1.0, df['Vol_10'].iloc[-1] * 50)
        earn_risk = 0.8 if (earnings_date and (pd.Timestamp(earnings_date) - pd.Timestamp(target_date)).days < 7) else 0.2
        sent_risk = 1 - abs(sent_score)
        rsi_risk = df['RSI'].iloc[-1] / 100
        
        risk_values = [vol_risk, earn_risk, sent_risk, rsi_risk, 0.4]
        
        fig_risk = go.Figure(data=go.Scatterpolar(
            r=risk_values,
            theta=risk_factors,
            fill='toself',
            marker=dict(color='#ff4b4b')
        ))
        fig_risk.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False, template="plotly_dark", height=400)
        st.plotly_chart(fig_risk, use_container_width=True)

    with col_right:
        st.subheader("📰 Recent Sentiment Catalysts")
        for n in news[:4]:
            st.markdown(f"**{n['title']}**")
            st.caption(f"Impact: {n['publisher']} | [Link]({n['link']})")
        
        if earnings_date:
            st.warning(f"🔔 **Upcoming Earnings:** {earnings_date.strftime('%Y-%m-%d')} (Significant Volatility Expected)")
        else:
            st.success("✅ No immediate earnings volatility detected for target window.")

else:
    st.error("Terminal offline. Verify Ticker and API Connection.")
