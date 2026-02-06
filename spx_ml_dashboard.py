import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

# --- Page Config ---
st.set_page_config(layout="wide", page_title="AI Alpha v9.5: Auditor", page_icon="⚖️")

# --- Core Data Engine ---
@st.cache_data(ttl=3600)
def get_auditor_data(ticker, horizon=5):
    df = yf.download(ticker, period="max", interval="1d")
    tnx = yf.download("^TNX", period="max", interval="1d")['Close']
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # Advanced Quant Features
    df['TNX'] = tnx
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_10'] = df['Log_Ret'].rolling(10).std()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    df['Target'] = df['Close'].shift(-horizon) / df['Close'] - 1
    return df.dropna()

data = get_auditor_data(st.sidebar.text_input("Ticker Symbol", "NVDA").upper())

if data is not None:
    tab_live, tab_audit = st.tabs(["⚡ Live Prediction", "🔬 Advanced Audit & Post-Mortem"])

    with tab_live:
        st.info("The Live Prediction tab operates as configured in v9.1. Navigate to 'Advanced Audit' for the new deep-dive tools.")

    with tab_audit:
        # --- 1. Audit Controls ---
        st.subheader("🧪 Historical Stress Test Parameters")
        audit_days = st.slider("Select Audit Period (Days Back)", 60, 730, 250)
        test_df = data.tail(audit_days).copy()
        features = ['RSI', 'Vol_10', 'TNX']
        
        # --- 2. Advanced Backtest Engine ---
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(test_df[features].values, test_df['Target'].values * 100)
        
        # Simulated Trading Logic
        test_df['AI_Score'] = model.predict(test_df[features].values)
        test_df['Signal'] = np.where(test_df['AI_Score'] > 0.5, 1, 0) # Buy if AI predicts > 0.5% gain
        test_df['Strategy_Ret'] = test_df['Signal'].shift(1) * test_df['Log_Ret']
        test_df['Strat_Cum'] = np.exp(test_df['Strategy_Ret'].cumsum())
        test_df['Mkt_Cum'] = np.exp(test_df['Log_Ret'].cumsum())
        
        # Quant Metrics
        total_trades = test_df['Signal'].diff().abs().sum() / 2
        win_rate = (test_df[test_df['Strategy_Ret'] > 0].shape[0] / test_df[test_df['Strategy_Ret'] != 0].shape[0]) * 100
        sharpe = (test_df['Strategy_Ret'].mean() / test_df['Strategy_Ret'].std()) * np.sqrt(252)
        mdd = ((test_df['Strat_Cum'] / test_df['Strat_Cum'].cummax()) - 1).min() * 100

        # --- 3. Metric Dashboard ---
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Strategy Alpha", f"{(test_df['Strat_Cum'].iloc[-1] - test_df['Mkt_Cum'].iloc[-1])*100:+.2f}%")
        c2.metric("Win Probability", f"{win_rate:.1f}%")
        c3.metric("Sharpe Ratio", f"{sharpe:.2f}")
        c4.metric("Max Pain (MDD)", f"{mdd:.1f}%")

        st.divider()

        # --- 4. Decision Subplots ---
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            st.write("### 📈 Equity Growth vs. Market Benchmark")
            fig_equity = go.Figure()
            fig_equity.add_trace(go.Scatter(x=test_df.index, y=test_df['Strat_Cum'], name="AI Strategy", line=dict(color="#00ffcc", width=3)))
            fig_equity.add_trace(go.Scatter(x=test_df.index, y=test_df['Mkt_Cum'], name="Buy & Hold", line=dict(color="#666", dash='dot')))
            st.plotly_chart(fig_equity.update_layout(template="plotly_dark", height=400), use_container_width=True)
            st.caption(f"**Interpretation:** This graph tracks a hypothetical $1 investment over the last **{audit_days} days**. The green line is the AI's actively managed path. If the green line is above the gray line, the AI has generated **Alpha**.")

        with col_right:
            st.write("### 🧬 AI Decision Logic (Feature Weight)")
            importances = model.feature_importances_
            fig_feat = go.Figure(go.Bar(x=features, y=importances, marker_color='#ff9500'))
            st.plotly_chart(fig_feat.update_layout(template="plotly_dark", height=400), use_container_width=True)
            st.caption(f"**Interpretation:** During this **{audit_days}-day** period, the AI found **{features[np.argmax(importances)]}** to be the most reliable predictor of price movement.")

        st.divider()

        # --- 5. Volatility & Underwater Analysis ---
        st.write("### 🕳️ Max Pain & Drawdown Analysis")
        drawdown = (test_df['Strat_Cum'] / test_df['Strat_Cum'].cummax()) - 1
        fig_dd = go.Figure(go.Scatter(x=test_df.index, y=drawdown*100, fill='tozeroy', line=dict(color='#ff4b4b')))
        st.plotly_chart(fig_dd.update_layout(template="plotly_dark", height=250, title="Underwater Plot (%)"), use_container_width=True)
        st.caption(f"**Interpretation:** This chart shows the 'dips.' Every time the red area drops, it represents a period where the AI was losing capital from its previous peak. Smaller red valleys mean a 'smoother' ride for the investor.")

        # --- 6. Trade Journal (Final Table) ---
        with st.expander("📝 Detailed Trade Journal"):
            journal = test_df[test_df['Signal'].diff() != 0][['Close', 'AI_Score', 'Signal']]
            st.dataframe(journal.tail(20), use_container_width=True)
