import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

# --- Page Config ---
st.set_page_config(layout="wide", page_title="AI Alpha Terminal v7.3", page_icon="🏛️")

# --- Global Sidebar ---
with st.sidebar:
    st.header("⚙️ Quant Lab")
    ticker = st.text_input("Ticker Symbol", "NVDA").upper()
    forecast_days = st.slider("Forecast Horizon (Days)", 1, 10, 5)
    
    st.divider()
    st.caption("v7.3: Restored UI & Quant Features")
    vol_threshold = st.slider("Volatility Safety Cutoff (%)", 50, 100, 85)

# --- Core Processing Logic ---
@st.cache_data(ttl=3600)
def get_data_with_lags(ticker, horizon=5):
    df = yf.download(ticker, period="10y", interval="1d")
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Technical Indicators
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_10'] = df['Log_Ret'].rolling(10).std()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['STD20'] = df['Close'].rolling(20).std()
    df['BB_Up'] = df['MA20'] + (df['STD20'] * 2)
    df['BB_Low'] = df['MA20'] - (df['STD20'] * 2)
    
    # Memory Lags
    df['Ret_Lag1'] = df['Log_Ret'].shift(1)
    df['Ret_Lag2'] = df['Log_Ret'].shift(2)
    df['Vol_Lag1'] = df['Vol_10'].shift(1)
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    # Target
    df['Target'] = df['Close'].shift(-horizon) / df['Close'] - 1
    return df.dropna()

def train_model(df):
    features = ['RSI', 'Vol_10', 'Ret_Lag1', 'Ret_Lag2', 'Vol_Lag1']
    X = df[features].values
    y = df['Target'].values * 100
    model = RandomForestRegressor(n_estimators=150, max_depth=7, min_samples_leaf=10, random_state=42)
    model.fit(X, y)
    return model, features

data = get_data_with_lags(ticker, horizon=forecast_days)

if data is not None:
    tab_live, tab_audit = st.tabs(["⚡ Live Prediction", "🧪 Custom Range Audit"])

    # --- TAB 1: LIVE TERMINAL (Restored UI) ---
    with tab_live:
        model, feature_names = train_model(data)
        last_row = data.iloc[-1]
        
        # Logic Calculations
        current_vol = last_row['Vol_10']
        vol_cutoff = data['Vol_10'].quantile(vol_threshold/100)
        is_safe = current_vol < vol_cutoff
        
        input_data = last_row[feature_names].values.reshape(1, -1)
        pred_return = model.predict(input_data)[0]
        
        # Decision UI
        if not is_safe:
            status, color, advice = "CASH (DANGER)", "#ff4b4b", "Volatility spike detected. Preservation mode active."
        else:
            status = "STRONG BUY" if pred_return > 1.5 else "BUY" if pred_return > 0.5 else "SELL" if pred_return < -0.5 else "NEUTRAL"
            color = "#00ffcc" if "BUY" in status else "#ff4b4b" if "SELL" in status else "#ff9500"
            advice = f"AI suggests a {status} stance for the next {forecast_days} days."

        st.markdown(f"""
            <div style='background-color:{color}22; border:2px solid {color}; padding:25px; border-radius:15px; text-align:center; margin-bottom:20px;'>
                <h1 style='color:{color}; margin:0;'>{status}</h1>
                <h3 style='margin:5px 0;'>Forecasted Move: {pred_return:.2f}%</h3>
                <p>{advice}</p>
            </div>
        """, unsafe_allow_html=True)

        col_left, col_right = st.columns([3, 1])
        
        with col_left:
            # Independent Price Chart
            fig_p = go.Figure()
            fig_p.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="Price"))
            fig_p.add_trace(go.Scatter(x=data.index, y=data['BB_Up'], line=dict(color='rgba(255,255,255,0.2)'), name="BB Up"))
            fig_p.add_trace(go.Scatter(x=data.index, y=data['BB_Low'], line=dict(color='rgba(255,255,255,0.2)'), fill='tonexty', fillcolor='rgba(0,255,204,0.05)', name="BB Low"))
            fig_p.update_layout(title="Price Action & Volatility Bands", template="plotly_dark", height=450, xaxis_range=[data.index[-90], data.index[-1]], margin=dict(t=30, b=0))
            fig_p.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
            st.plotly_chart(fig_p, use_container_width=True)

            # Independent RSI Chart
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI'], line=dict(color='orange', width=2), name="RSI"))
            fig_rsi.add_hline(y=70, line_dash="dot", line_color="red")
            fig_rsi.add_hline(y=30, line_dash="dot", line_color="green")
            fig_rsi.update_layout(title="RSI Momentum Detector", template="plotly_dark", height=200, xaxis_range=[data.index[-90], data.index[-1]], margin=dict(t=30, b=30))
            fig_rsi.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
            st.plotly_chart(fig_rsi, use_container_width=True)

        with col_right:
            st.metric("Expected Price", f"${data['Close'].iloc[-1] * (1 + pred_return/100):,.2f}")
            st.metric("Market Volatility", f"{current_vol*100:.2f}%", f"{'HIGH' if not is_safe else 'NORMAL'}", delta_color="inverse")
            st.divider()
            st.write("### 🧠 AI Logic Weight")
            importances = pd.DataFrame({'Feature': feature_names, 'Weight': model.feature_importances_}).sort_values(by='Weight')
            st.bar_chart(importances.set_index('Feature'), color=color, horizontal=True)

    # --- TAB 2: AUDIT (Range-Based) ---
    with tab_audit:
        st.subheader("🕵️ Era-Specific Walk-Forward Audit")
        c1, c2 = st.columns(2)
        audit_start = c1.date_input("Audit Start", value=data.index.max() - timedelta(days=365))
        audit_end = c2.date_input("Audit End", value=data.index.max())
        
        if st.button("🚀 Run Range Simulation"):
            t_start, t_end = pd.Timestamp(audit_start), pd.Timestamp(audit_end)
            sim_data = data[(data.index >= t_start) & (data.index <= t_end)].copy()
            
            # Logic for Walk-Forward
            strat_equity = [10000]; mkt_equity = [10000]; dates = []
            pre_sim_data = data[data.index < t_start]
            dynamic_cutoff = pre_sim_data['Vol_10'].quantile(vol_threshold/100)
            
            for i in range(0, len(sim_data) - 5, 5):
                curr_date = sim_data.index[i]
                train_sub = data[data.index < curr_date]
                m, feats = train_model(train_sub)
                
                # Trade Decision
                curr_vol = sim_data.iloc[i]['Vol_10']
                if curr_vol > dynamic_cutoff:
                    realized = 0.0
                else:
                    X_test = sim_data.iloc[i][feats].values.reshape(1, -1)
                    pred = m.predict(X_test)[0]
                    realized = sim_data.iloc[i]['Target'] * 100 if pred > 0 else 0.0

                strat_equity.append(strat_equity[-1] * (1 + realized/100))
                mkt_equity.append(mkt_equity[-1] * (1 + (sim_data.iloc[i]['Target'] * 100)/100))
                dates.append(curr_date)

            fig_aud = go.Figure()
            fig_aud.add_trace(go.Scatter(x=dates, y=strat_equity[1:], name="AI Strategy", line=dict(color='#00ffcc')))
            fig_aud.add_trace(go.Scatter(x=dates, y=mkt_equity[1:], name="Market", line=dict(color='gray', dash='dash')))
            fig_aud.update_layout(template="plotly_dark", title="Audit Result")
            st.plotly_chart(fig_aud, use_container_width=True)
else:
    st.error("Invalid Ticker.")


