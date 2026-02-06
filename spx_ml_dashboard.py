import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

# Robust Sentiment Import (Prevents Crash)
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    nltk.data.find('vader_lexicon')
    sia = SentimentIntensityAnalyzer()
    HAS_SENTIMENT = True
except:
    HAS_SENTIMENT = False

# --- Page Config ---
st.set_page_config(layout="wide", page_title="AI Alpha v9.1: Quant-Commander", page_icon="🕵️‍♂️")

# --- Global Sidebar ---
with st.sidebar:
    st.header("⚙️ Command Center")
    ticker = st.text_input("Ticker Symbol", "NVDA").upper()
    forecast_days = st.slider("Forecast Horizon (Days)", 1, 10, 5)
    st.divider()
    timeframes = {"1W": 7, "1M": 30, "3M": 90, "6M": 180, "1Y": 365, "2Y": 730}
    selected_tf = st.selectbox("Viewport Range", list(timeframes.keys()), index=2)
    lookback = timeframes[selected_tf]

# --- Core Intelligence Engine ---
@st.cache_data(ttl=3600)
def get_master_data(ticker, horizon=5):
    df = yf.download(ticker, period="max", interval="1d")
    tnx = yf.download("^TNX", period="max", interval="1d")['Close']
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # Feature Engineering
    df['TNX'] = tnx
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_10'] = df['Log_Ret'].rolling(10).std()
    
    # RSI Calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    # Shifted Target for AI Training
    df['Target'] = df['Close'].shift(-horizon) / df['Close'] - 1
    return df.dropna()

data = get_master_data(ticker, forecast_days)

if data is not None:
    tab_live, tab_audit = st.tabs(["⚡ Live Prediction", "🧪 Custom Range Audit"])

    with tab_live:
        # --- AI INFERENCE ---
        features = ['RSI', 'Vol_10', 'TNX']
        model = RandomForestRegressor(n_estimators=100, random_state=42).fit(data[features].values, data['Target'].values * 100)
        last_row = data.iloc[-1]
        pred_move = model.predict(last_row[features].values.reshape(1, -1))[0]
        
        # News Sentiment
        sent_val = 0
        if HAS_SENTIMENT:
            try:
                news = yf.Ticker(ticker).news
                sent_val = np.mean([sia.polarity_scores(n['title'])['compound'] for n in news[:5]])
            except: pass

        # --- VISUAL 1: PROBABILITY CONE ---
        vdf = data.tail(lookback)
        fut_dates = [vdf.index[-1] + timedelta(days=i) for i in range(forecast_days + 1)]
        
        # Monte Carlo 100 Paths
        mu, sigma = data['Log_Ret'].mean(), data['Log_Ret'].std()
        sims = np.array([[last_row['Close'] * np.exp(np.cumsum(np.random.normal(mu, sigma, forecast_days)))] for _ in range(100)]).reshape(100, forecast_days)
        sims = np.insert(sims, 0, last_row['Close'], axis=1)
        
        fig_cone = go.Figure()
        fig_cone.add_trace(go.Candlestick(x=vdf.index, open=vdf['Open'], high=vdf['High'], low=vdf['Low'], close=vdf['Close'], name="History"))
        fig_cone.add_trace(go.Scatter(x=fut_dates, y=np.percentile(sims, 90, axis=0), line=dict(width=0), showlegend=False))
        fig_cone.add_trace(go.Scatter(x=fut_dates, y=np.percentile(sims, 10, axis=0), fill='tonexty', fillcolor='rgba(0, 255, 204, 0.1)', line=dict(width=0), name="90% Confidence Cone"))
        fig_cone.add_trace(go.Scatter(x=fut_dates, y=[last_row['Close'] * (1 + (pred_move/100 * (i/forecast_days))) for i in range(forecast_days+1)], 
                                      line=dict(color='#00ffcc', dash='dot', width=3), name="AI Target"))
        
        st.plotly_chart(fig_cone.update_layout(template="plotly_dark", height=450, xaxis_rangeslider_visible=False), use_container_width=True)
        st.info(f"**Interpretation:** Over the last **{selected_tf}**, we track price action. The **Shaded Cone** represents 100 simulated futures. The **Dotted Line** is the AI's specific target of **{pred_move:+.2f}%**.")

        # --- VISUAL 2: VOLUME & RSI ---
        c1, c2 = st.columns(2)
        with c1:
            v_colors = ['#00ffcc' if vdf['Close'].iloc[i] >= vdf['Open'].iloc[i] else '#ff4b4b' for i in range(len(vdf))]
            st.plotly_chart(go.Figure(go.Bar(x=vdf.index, y=vdf['Volume'], marker_color=v_colors)).update_layout(template="plotly_dark", height=250, title="Order Flow (Volume)"), use_container_width=True)
            st.warning("**Interpretation:** High volume bars confirm the AI move. If the AI predicts a rally but Volume is shrinking, stay cautious.")
        with c2:
            fig_rsi = go.Figure(go.Scatter(x=vdf.index, y=vdf['RSI'], line=dict(color='orange', width=2)))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
            st.plotly_chart(fig_rsi.update_layout(template="plotly_dark", height=250, title="Momentum RSI (Orange Line)", yaxis=dict(range=[0,100])), use_container_width=True)
            st.success("**Interpretation:** RSI measures 'heat'. Crossing the **Red Line** means the rally is exhausted (Overbought).")

    with tab_audit:
        st.subheader("🧪 Tactical Stress Test (Backtest)")
        audit_days = st.slider("Audit Window (Days)", 30, 365, 180)
        test_df = data.tail(audit_days).copy()
        
        # Backtest Logic: Long if AI predicts > 0.5%
        test_df['Signal'] = (model.predict(test_df[features].values) > 0.5).astype(int)
        test_df['Strategy_Ret'] = test_df['Signal'].shift(1) * test_df['Log_Ret']
        test_df['Strat_Cum'] = np.exp(test_df['Strategy_Ret'].cumsum())
        test_df['Mkt_Cum'] = np.exp(test_df['Log_Ret'].cumsum())
        
        # Max Drawdown
        test_df['Peak'] = test_df['Strat_Cum'].cummax()
        test_df['Drawdown'] = (test_df['Strat_Cum'] / test_df['Peak']) - 1
        mdd = test_df['Drawdown'].min() * 100

        # Performance Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Strategy Return", f"{(test_df['Strat_Cum'].iloc[-1]-1)*100:.1f}%")
        m2.metric("Buy & Hold", f"{(test_df['Mkt_Cum'].iloc[-1]-1)*100:.1f}%")
        m3.metric("Max Pain (Drawdown)", f"{mdd:.1f}%", delta_color="inverse")

        # Visual Audit Chart
        fig_audit = go.Figure()
        fig_audit.add_trace(go.Scatter(x=test_df.index, y=test_df['Strat_Cum'], name="AI Strategy", line=dict(color="#00ffcc")))
        fig_audit.add_trace(go.Scatter(x=test_df.index, y=test_df['Mkt_Cum'], name="Buy & Hold", line=dict(color="gray", dash='dot')))
        st.plotly_chart(fig_audit.update_layout(template="plotly_dark", height=400, title=f"Equity Growth: Last {audit_days} Days"), use_container_width=True)
        
        st.info(f"**Audit Breakdown:** This chart simulates trading {ticker} based on AI signals for the last **{audit_days} days**. "
                f"We calculate 'Max Pain' by tracking the deepest dip from the strategy's peak value. If the AI line is above the dotted line, "
                f"it's beating the market.")
