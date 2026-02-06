import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime, timedelta

# --- VADER Import ---
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

# --- Core Setup ---
st.set_page_config(layout="wide", page_title="AI Alpha Terminal v5.4", page_icon="🏛️")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Terminal Config")
    ticker = st.text_input("Ticker Symbol", "NVDA").upper()
    target_dt = st.date_input("Forecast Target Date", datetime.now() + timedelta(days=7))
    st.divider()
    st.info("The terminal uses a 0.5% Truth-Threshold and TimeSeriesSplit for data integrity.")

# --- Data Functions ---
@st.cache_data(ttl=3600)
def get_processed_data(ticker):
    df = yf.download(ticker, period="5y", interval="1d")
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Technical Indicators
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
    return df.dropna()

def get_sentiment(ticker):
    if not VADER_AVAILABLE: return 0
    try:
        tk = yf.Ticker(ticker)
        news = tk.news[:5]
        analyzer = SentimentIntensityAnalyzer()
        scores = [analyzer.polarity_scores(n['title'])['compound'] for n in news]
        return np.mean(scores) if scores else 0
    except: return 0

# --- MAIN APP ---
data_full = get_processed_data(ticker)

if data_full is not None:
    tab_live, tab_audit = st.tabs(["🏛️ Live Terminal", "🕵️ Historical Audit"])

    # --- TAB 1: RESTORED LIVE TERMINAL ---
    with tab_live:
        # Prepare ML Model
        features = ['RSI', 'Vol_10', 'Log_Ret']
        X = data_full[features].values
        y = data_full['Log_Ret'].values
        scaler = StandardScaler().fit(X)
        X_s = scaler.transform(X)
        
        # Win Rate Calculation (TimeSeriesSplit)
        tscv = TimeSeriesSplit(n_splits=5)
        win_rates = []
        for tr_idx, te_idx in tscv.split(X_s):
            m_fold = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
            m_fold.fit(X_s[tr_idx], y[tr_idx])
            preds = m_fold.predict(X_s[te_idx])
            wins = (np.sign(preds) == np.sign(y[te_idx])) & (np.abs(y[te_idx]) >= 0.005)
            win_rates.append(wins.mean())
        
        true_win_rate = np.mean(win_rates) * 100
        
        # Production Model & Prediction
        model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_s, y)
        last_state = X_s[-1:]
        pred_ret = model.predict(last_state)[0]
        days_out = (pd.Timestamp(target_dt) - data_full.index[-1]).days
        final_p = data_full['Close'].iloc[-1] * np.exp(pred_ret * days_out)
        proj_move = ((final_p / data_full['Close'].iloc[-1]) - 1) * 100
        sent_score = get_sentiment(ticker)

        # Decision Engine Logic
        signals = {"BUY": 0, "SELL": 0, "HOLD": 0}
        reasons = []
        rsi_now = data_full['RSI'].iloc[-1]
        price_now = data_full['Close'].iloc[-1]

        if rsi_now < 35: signals["BUY"] += 1; reasons.append("RSI: Oversold")
        elif rsi_now > 65: signals["SELL"] += 1; reasons.append("RSI: Overbought")
        else: signals["HOLD"] += 1; reasons.append("RSI: Neutral")

        if price_now < data_full['BB_Low'].iloc[-1]: signals["BUY"] += 1; reasons.append("BBands: Below Lower")
        elif price_now > data_full['BB_Up'].iloc[-1]: signals["SELL"] += 1; reasons.append("BBands: Above Upper")
        else: signals["HOLD"] += 1; reasons.append("BBands: Within Range")

        if proj_move > 0.5: signals["BUY"] += 1; reasons.append(f"AI: Bullish (+{proj_move:.1f}%)")
        elif proj_move < -0.5: signals["SELL"] += 1; reasons.append(f"AI: Bearish ({proj_move:.1f}%)")
        else: signals["HOLD"] += 1; reasons.append("AI: Neutral Forecast")

        decision = "BUY" if signals["BUY"] >= 2 else "SELL" if signals["SELL"] >= 2 else "HOLD"
        color = "#00ffcc" if decision == "BUY" else "#ff4b4b" if decision == "SELL" else "#ff9500"

        # Display Signal Header
        st.markdown(f"""
            <div style="background-color:{color}22; border: 2px solid {color}; padding: 25px; border-radius: 12px; text-align: center; margin-bottom: 20px;">
                <h1 style="color:{color}; margin:0;">SIGNAL: {decision}</h1>
                <p style="color:white; font-size:18px; margin:5px;"><b>Reasoning:</b> {', '.join(reasons)}</p>
            </div>
        """, unsafe_allow_html=True)

        # Main Layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=data_full.index[-120:], open=data_full['Open'], high=data_full['High'], low=data_full['Low'], close=data_full['Close'], name="Price"), row=1, col=1)
            fig.add_trace(go.Scatter(x=data_full.index[-120:], y=data_full['BB_Up'].iloc[-120:], line=dict(color='rgba(255,255,255,0.2)'), name="Upper BB"), row=1, col=1)
            fig.add_trace(go.Scatter(x=data_full.index[-120:], y=data_full['BB_Low'].iloc[-120:], line=dict(color='rgba(255,255,255,0.2)'), fill='tonexty', name="Lower BB"), row=1, col=1)
            fig.add_trace(go.Scatter(x=data_full.index[-120:], y=data_full['RSI'].iloc[-120:], line=dict(color='orange'), name="RSI"), row=2, col=1)
            fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.metric("Truth-Tested Win Rate", f"{win_rate:.1f}%")
            st.metric("7-Day Target Price", f"${final_p:,.2f}")
            st.metric("News Sentiment", f"{sent_score:.2f}")
            
            st.subheader("📋 Data Definitions")
            st.caption("**Win Rate:** Based on TimeSeriesSplit where actual move > 0.5%.")
            st.caption("**Signal:** Requires 2 out of 3 pillars (RSI, BB, AI) to agree.")
            st.latex(r"Acc = \frac{Hits \cap |Ret| > 0.5\%}{Total}")

    # --- TAB 2: AUDIT TERMINAL ---
    with tab_audit:
        st.subheader("🕵️ Custom Historical Audit")
        c1, c2 = st.columns(2)
        a_start = c1.date_input("Audit Start", datetime.now() - timedelta(days=200))
        a_end = c2.date_input("Audit End", datetime.now() - timedelta(days=20))
        
        if st.button("Generate Historical Audit Report"):
            # Split data for honest testing
            audit_full = data_full[:pd.Timestamp(a_end)]
            train_box = data_full[:pd.Timestamp(a_start)]
            test_box = audit_full[pd.Timestamp(a_start):]
            
            if len(train_box) > 100:
                scaler_a = StandardScaler().fit(train_box[features])
                m_audit = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
                m_audit.fit(scaler_a.transform(train_box[features]), train_box['Log_Ret'])
                
                # Run Test
                a_preds = m_audit.predict(scaler_a.transform(test_box[features]))
                a_hits = (np.sign(a_preds) == np.sign(test_box['Log_Ret'])) & (np.abs(test_box['Log_Ret']) > 0.005)
                
                st.metric("Audit Accuracy", f"{a_hits.mean()*100:.1f}%")
                
                # Equity Curve
                test_box = test_box.copy()
                test_box['Strategy_Ret'] = np.sign(a_preds) * test_box['Log_Ret']
                test_box['Equity'] = (1 + test_box['Strategy_Ret']).cumprod() * 10000
                
                fig_a = go.Figure(go.Scatter(x=test_box.index, y=test_box['Equity'], name="Strategy Growth", line=dict(color="#00ffcc")))
                fig_a.update_layout(title="Growth of $10,000 in Audit Window", template="plotly_dark", height=400)
                st.plotly_chart(fig_a, use_container_width=True)
            else:
                st.error("Insufficient history before the audit start date.")
else:
    st.error("Data fetch failed. Verify ticker.")

