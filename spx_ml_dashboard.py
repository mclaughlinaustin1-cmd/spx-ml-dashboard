import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

# --- Page Config ---
st.set_page_config(layout="wide", page_title="AI Alpha Terminal v7.7", page_icon="🏛️")

# --- Global Sidebar ---
with st.sidebar:
    st.header("⚙️ Quant Lab")
    ticker = st.text_input("Ticker Symbol", "NVDA").upper()
    forecast_days = st.slider("Forecast Horizon (Days)", 1, 10, 5)
    
    st.divider()
    st.subheader("🛡️ Risk Management")
    vol_threshold = st.slider("Volatility Safety Cutoff (%)", 50, 100, 85, 
                              help="Top percentile of volatility where AI stays in Cash.")
    
    st.divider()
    st.subheader("📅 Display Settings")
    timeframes = {
        "1 Week": 7, "14 Days": 14, "30 Days": 30, "60 Days": 60, "90 Days": 90,
        "6 Months": 180, "1 Year": 365, "2 Years": 730, "5 Years": 1825, "10 Years": 3650, "Max": 10000
    }
    selected_tf = st.selectbox("Live Chart View", list(timeframes.keys()), index=4)

# --- Core Data Engine ---
@st.cache_data(ttl=3600)
def get_data(ticker, horizon=5):
    # Fetch max data to support all timeframes and training
    df = yf.download(ticker, period="max", interval="1d")
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Feature Engineering
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_10'] = df['Log_Ret'].rolling(10).std()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['STD20'] = df['Close'].rolling(20).std()
    df['BB_Up'] = df['MA20'] + (df['STD20'] * 2)
    df['BB_Low'] = df['MA20'] - (df['STD20'] * 2)
    
    # Lags (AI Memory)
    df['Ret_Lag1'] = df['Log_Ret'].shift(1)
    df['Ret_Lag2'] = df['Log_Ret'].shift(2)
    df['Vol_Lag1'] = df['Vol_10'].shift(1)
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    # Target: Percent move 'horizon' days into the future
    df['Target'] = df['Close'].shift(-horizon) / df['Close'] - 1
    return df.dropna()

def train_model(df):
    features = ['RSI', 'Vol_10', 'Ret_Lag1', 'Ret_Lag2', 'Vol_Lag1']
    X = df[features].values
    y = df['Target'].values * 100
    model = RandomForestRegressor(n_estimators=150, max_depth=7, min_samples_leaf=10, random_state=42)
    model.fit(X, y)
    return model, features

data = get_data(ticker, horizon=forecast_days)

if data is not None:
    tab_live, tab_audit = st.tabs(["⚡ Live Prediction", "🧪 Custom Range Audit"])

    # --- TAB 1: LIVE TERMINAL ---
    with tab_live:
        model, feature_names = train_model(data)
        last_row = data.iloc[-1]
        
        # Decision Logic
        current_vol = last_row['Vol_10']
        vol_cutoff = data['Vol_10'].quantile(vol_threshold/100)
        is_safe = current_vol < vol_cutoff
        
        input_vec = last_row[feature_names].values.reshape(1, -1)
        pred_move = model.predict(input_vec)[0]
        
        # Banner UI
        if not is_safe:
            status, color = "PROTECTIVE CASH", "#ff4b4b"
            advice = "Volatility is too high. AI has moved to safety."
        else:
            status = "BULLISH" if pred_move > 0.5 else "BEARISH" if pred_move < -0.5 else "NEUTRAL"
            color = "#00ffcc" if status == "BULLISH" else "#ff4b4b" if status == "BEARISH" else "#ff9500"
            advice = f"Market regime is stable. AI expects a {pred_move:.2f}% move."

        st.markdown(f"""
            <div style='background-color:{color}22; border:2px solid {color}; padding:20px; border-radius:15px; text-align:center; margin-bottom:20px'>
                <h1 style='color:{color}; margin:0;'>{status}</h1>
                <p style='font-size:1.2em; margin:5px 0;'>{advice}</p>
            </div>
        """, unsafe_allow_html=True)

        col_main, col_metrics = st.columns([3, 1])
        
        with col_main:
            # Dynamic Slicing & Auto-Scaling
            lookback = timeframes[selected_tf]
            vdf = data.tail(lookback)
            y_min, y_max = vdf['Low'].min() * 0.97, vdf['High'].max() * 1.03
            
            # Subplot: Price + Volume
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
            
            # Candlesticks
            fig.add_trace(go.Candlestick(x=vdf.index, open=vdf['Open'], high=vdf['High'], low=vdf['Low'], close=vdf['Close'], name="Price"), row=1, col=1)
            # BB Bands
            fig.add_trace(go.Scatter(x=vdf.index, y=vdf['BB_Up'], line=dict(color='rgba(255,255,255,0.1)'), name="Upper Band"), row=1, col=1)
            fig.add_trace(go.Scatter(x=vdf.index, y=vdf['BB_Low'], line=dict(color='rgba(255,255,255,0.1)'), name="Lower Band"), row=1, col=1)
            
            # Color-coded Volume
            v_colors = ['#00ffcc' if vdf['Close'].iloc[i] >= vdf['Open'].iloc[i] else '#ff4b4b' for i in range(len(vdf))]
            fig.add_trace(go.Bar(x=vdf.index, y=vdf['Volume'], marker_color=v_colors, name="Volume"), row=2, col=1)

            fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False, 
                              yaxis1=dict(range=[y_min, y_max], fixedrange=False), margin=dict(t=20, b=0))
            st.plotly_chart(fig, use_container_width=True)

            # RSI Chart
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=vdf.index, y=vdf['RSI'], line=dict(color='orange', width=2)))
            fig_rsi.add_hline(y=70, line_dash="dot", line_color="#ff4b4b")
            fig_rsi.add_hline(y=30, line_dash="dot", line_color="#00ffcc")
            fig_rsi.update_layout(template="plotly_dark", height=200, title="RSI Momentum Detector", margin=dict(t=30, b=20))
            st.plotly_chart(fig_rsi, use_container_width=True)

        with col_metrics:
            st.metric("Last Close", f"${last_row['Close']:,.2f}")
            st.metric("Forecasted Move", f"{pred_move:.2f}%")
            st.write("---")
            st.write("### AI Feature Importance")
            importances = pd.DataFrame({'W': model.feature_importances_}, index=feature_names).sort_values('W')
            st.bar_chart(importances, horizontal=True)

    # --- TAB 2: CUSTOM RANGE AUDIT ---
    with tab_audit:
        st.subheader("🧪 Targeted Historical Stress Test")
        c1, c2 = st.columns(2)
        sim_start = c1.date_input("Simulation Start Date", value=datetime.now() - timedelta(days=365))
        sim_end = c2.date_input("Simulation End Date", value=datetime.now())
        
        if st.button("🚀 Run Backtest"):
            t_start, t_end = pd.Timestamp(sim_start), pd.Timestamp(sim_end)
            sim_data = data[(data.index >= t_start) & (data.index <= t_end)].copy()
            
            if len(sim_data) < 10:
                st.error("Select a wider date range.")
            else:
                # Walk-forward Logic
                strat_vals, mkt_vals, dates = [10000], [10000], []
                progress = st.progress(0)
                
                for i in range(0, len(sim_data)-5, 5):
                    curr_dt = sim_data.index[i]
                    # Retrain only on data before the 'current' sim date
                    train_set = data[data.index < curr_dt]
                    m, fts = train_model(train_set)
                    
                    row = sim_data.iloc[i]
                    actual_move = row['Target'] * 100
                    
                    # Safety Switch
                    dyn_cutoff = train_set['Vol_10'].quantile(vol_threshold/100)
                    if row['Vol_10'] > dyn_cutoff:
                        realized = 0.0 # Safety
                    else:
                        p = m.predict(row[fts].values.reshape(1, -1))[0]
                        realized = actual_move if p > 0 else 0.0
                    
                    strat_vals.append(strat_vals[-1] * (1 + realized/100))
                    mkt_vals.append(mkt_vals[-1] * (1 + actual_move/100))
                    dates.append(curr_dt)
                    progress.progress(i / len(sim_data))
                
                progress.empty()
                
                # Visuals
                fig_audit = go.Figure()
                fig_audit.add_trace(go.Scatter(x=dates, y=strat_vals[1:], name="AI Strategy", line=dict(color="#00ffcc")))
                fig_audit.add_trace(go.Scatter(x=dates, y=mkt_vals[1:], name="Market (B&H)", line=dict(color="gray", dash='dash')))
                fig_audit.update_layout(template="plotly_dark", title="Audit Results: Equity Curve")
                st.plotly_chart(fig_audit, use_container_width=True)
                
                final_ret = ((strat_vals[-1]/10000)-1)*100
                st.metric("Total Strategy Return", f"{final_ret:.2f}%")

else:
    st.error("Symbol not found or data error.")
