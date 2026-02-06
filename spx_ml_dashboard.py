import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

# --- NLTK Guard for Sentiment ---
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    try:
        nltk.data.find('vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()
    HAS_SENTIMENT = True
except:
    HAS_SENTIMENT = False

# --- Page Config ---
st.set_page_config(layout="wide", page_title="AI Alpha v9.6", page_icon="🏛️")

# --- Global Sidebar ---
with st.sidebar:
    st.header("⚙️ Quant Lab")
    ticker = st.text_input("Ticker Symbol", "NVDA").upper()
    forecast_days = st.slider("Forecast Horizon (Days)", 1, 14, 5)
    st.divider()
    vol_threshold = st.slider("Volatility Safety Cutoff (%)", 50, 100, 85)
    timeframes = {"1W": 7, "1M": 30, "3M": 90, "6M": 180, "1Y": 365, "2Y": 730}
    selected_tf = st.selectbox("Market Viewport", list(timeframes.keys()), index=2)
    lookback = timeframes[selected_tf]

# --- Core Data Engine ---
@st.cache_data(ttl=3600)
def get_terminal_data(ticker, horizon=5):
    df = yf.download(ticker, period="max", interval="1d")
    tnx = yf.download("^TNX", period="max", interval="1d")['Close']
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # Feature Engineering
    df['TNX'] = tnx
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_10'] = df['Log_Ret'].rolling(10).std()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    df['Target'] = df['Close'].shift(-horizon) / df['Close'] - 1
    return df.dropna()

data = get_terminal_data(ticker, forecast_days)

if data is not None:
    tab_live, tab_audit = st.tabs(["⚡ Live Prediction", "🧪 Advanced Audit"])

    # --- TAB 1: LIVE PREDICTION (v9.1 UI RESTORED) ---
    with tab_live:
        # AI Inference
        features = ['RSI', 'Vol_10', 'TNX']
        model = RandomForestRegressor(n_estimators=100, random_state=42).fit(data[features].values, data['Target'].values * 100)
        last_row = data.iloc[-1]
        pred_move = model.predict(last_row[features].values.reshape(1, -1))[0]
        
        # Sentiment Pulse
        sent_val = 0
        if HAS_SENTIMENT:
            try:
                news = yf.Ticker(ticker).news
                sent_val = np.mean([sia.polarity_scores(n['title'])['compound'] for n in news[:5]])
            except: pass

        # Signal Banner
        vol_limit = data['Vol_10'].quantile(vol_threshold/100)
        is_safe = last_row['Vol_10'] < vol_limit
        color = "#00ffcc" if pred_move > 0.5 and is_safe else "#ff4b4b" if not is_safe else "#ff9500"
        status = "BULLISH" if pred_move > 0.5 else "BEARISH" if pred_move < -0.5 else "NEUTRAL"
        
        st.markdown(f"<div style='background-color:{color}22; border:2px solid {color}; padding:20px; border-radius:15px; text-align:center;'><h1>{ticker}: {status} ({pred_move:+.2f}%)</h1></div>", unsafe_allow_html=True)

        col_main, col_stats = st.columns([3, 1])
        
        with col_main:
            # 1. PRICE & PROBABILITY FAN
            vdf = data.tail(lookback)
            fut_dates = [vdf.index[-1] + timedelta(days=i) for i in range(forecast_days + 1)]
            mu, sigma = data['Log_Ret'].mean(), data['Log_Ret'].std()
            sims = np.array([[last_row['Close'] * np.exp(np.cumsum(np.random.normal(mu, sigma, forecast_days)))] for _ in range(100)]).reshape(100, forecast_days)
            sims = np.insert(sims, 0, last_row['Close'], axis=1)
            
            fig_p = go.Figure()
            fig_p.add_trace(go.Candlestick(x=vdf.index, open=vdf['Open'], high=vdf['High'], low=vdf['Low'], close=vdf['Close'], name="History"))
            fig_p.add_trace(go.Scatter(x=fut_dates, y=np.percentile(sims, 90, axis=0), line=dict(width=0), showlegend=False))
            fig_p.add_trace(go.Scatter(x=fut_dates, y=np.percentile(sims, 10, axis=0), fill='tonexty', fillcolor='rgba(0, 255, 204, 0.1)', line=dict(width=0), name="90% Confidence Cone"))
            target_price = last_row['Close'] * (1 + (pred_move/100))
            fig_p.add_trace(go.Scatter(x=[vdf.index[-1], fut_dates[-1]], y=[last_row['Close'], target_price], line=dict(color=color, dash='dot', width=3), name="AI Target"))
            
            st.plotly_chart(fig_p.update_layout(template="plotly_dark", height=450, xaxis_rangeslider_visible=False), use_container_width=True)
            st.info(f"**Price Interpretation:** This chart shows the last **{selected_tf}** of action. The **Dotted Line** projects a target of **${target_price:.2f}** in {forecast_days} days. The **Shaded Cone** represents the statistical 'safety zone' based on 100 simulated futures.")

            # 2. VOLUME
            v_colors = ['#00ffcc' if vdf['Close'].iloc[i] >= vdf['Open'].iloc[i] else '#ff4b4b' for i in range(len(vdf))]
            st.plotly_chart(go.Figure(go.Bar(x=vdf.index, y=vdf['Volume'], marker_color=v_colors)).update_layout(template="plotly_dark", height=200, title="Order Flow (Volume)"), use_container_width=True)
            st.warning(f"**Volume Interpretation:** These bars show trading intensity over **{selected_tf}**. High volume (tall bars) during a predicted move confirms institutional backing. Low volume suggests a weak signal.")

            # 3. RSI
            fig_rsi = go.Figure(go.Scatter(x=vdf.index, y=vdf['RSI'], line=dict(color='orange', width=2)))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
            st.plotly_chart(fig_rsi.update_layout(template="plotly_dark", height=200, title="Momentum RSI", yaxis=dict(range=[0,100])), use_container_width=True)
            st.success(f"**Momentum Interpretation:** RSI measures the 'speed' of price. In this **{selected_tf}** view, a move above **70 (Red Line)** indicates the asset is overbought and due for a breather.")

        with col_stats:
            st.metric("AI Confidence", f"{pred_move:+.2f}%")
            st.metric("Sentiment", "Bullish" if sent_val > 0.05 else "Bearish" if sent_val < -0.05 else "Neutral")
            st.write("---")
            st.write("### Alpha Drivers")
            st.bar_chart(pd.DataFrame(model.feature_importances_, index=features, columns=['Weight']))
            # Macro Alert
            tnx_change = data['TNX'].pct_change(5).iloc[-1]
            if abs(tnx_change) > 0.03:
                st.error(f"Macro Alert: 10Y Yields shifted {tnx_change*100:+.1f}% this week.")

    # --- TAB 2: ADVANCED AUDIT (v9.5 LOGIC) ---
    with tab_audit:
        st.subheader("🕵️ Advanced Sovereign Audit")
        audit_window = st.slider("Audit Window (Days)", 30, 730, 250)
        adf = data.tail(audit_window).copy()
        
        # Simulation Logic
        adf['AI_Score'] = model.predict(adf[features].values)
        adf['Signal'] = np.where(adf['AI_Score'] > 0.5, 1, 0)
        adf['Strat_Ret'] = adf['Signal'].shift(1) * adf['Log_Ret']
        adf['Cum_Strat'] = np.exp(adf['Strat_Ret'].cumsum())
        adf['Cum_Mkt'] = np.exp(adf['Log_Ret'].cumsum())
        
        # Metrics
        m1, m2, m3 = st.columns(3)
        sharpe = (adf['Strat_Ret'].mean() / adf['Strat_Ret'].std()) * np.sqrt(252)
        mdd = ((adf['Cum_Strat'] / adf['Cum_Strat'].cummax()) - 1).min() * 100
        m1.metric("Strategy Alpha", f"{(adf['Cum_Strat'].iloc[-1] - adf['Cum_Mkt'].iloc[-1])*100:+.1f}%")
        m2.metric("Sharpe Ratio", f"{sharpe:.2f}")
        m3.metric("Max Pain (MDD)", f"{mdd:.1f}%")
        
        fig_audit = go.Figure()
        fig_audit.add_trace(go.Scatter(x=adf.index, y=adf['Cum_Strat'], name="AI Strategy", line=dict(color="#00ffcc", width=3)))
        fig_audit.add_trace(go.Scatter(x=adf.index, y=adf['Cum_Mkt'], name="Market", line=dict(color="gray", dash='dot')))
        st.plotly_chart(fig_audit.update_layout(template="plotly_dark", height=400, title="Equity Growth vs Market"), use_container_width=True)
        
        # Drawdown Plot
        dd = (adf['Cum_Strat'] / adf['Cum_Strat'].cummax()) - 1
        st.plotly_chart(go.Figure(go.Scatter(x=adf.index, y=dd*100, fill='tozeroy', line=dict(color='red'))).update_layout(template="plotly_dark", height=200, title="Underwater Analysis (%)"), use_container_width=True)
