import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

# --- Page Config ---
st.set_page_config(layout="wide", page_title="AI Alpha Terminal v7.8", page_icon="🏛️")

# --- Global Sidebar ---
with st.sidebar:
    st.header("⚙️ Quant Lab")
    ticker = st.text_input("Ticker Symbol", "NVDA").upper()
    forecast_days = st.slider("Forecast Horizon (Days)", 1, 10, 5)
    st.divider()
    vol_threshold = st.slider("Volatility Safety Cutoff (%)", 50, 100, 85)
    timeframes = {
        "1 Week": 7, "14 Days": 14, "30 Days": 30, "60 Days": 60, "90 Days": 90,
        "6 Months": 180, "1 Year": 365, "2 Years": 730, "5 Years": 1825, "10 Years": 3650
    }
    selected_tf = st.selectbox("Live Chart View", list(timeframes.keys()), index=4)

# --- Core Data Engine ---
@st.cache_data(ttl=3600)
def get_data(ticker, horizon=5):
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
    df['Ret_Lag1'] = df['Log_Ret'].shift(1)
    df['Ret_Lag2'] = df['Log_Ret'].shift(2)
    df['Vol_Lag1'] = df['Vol_10'].shift(1)
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    df['Target'] = df['Close'].shift(-horizon) / df['Close'] - 1
    return df.dropna()

def train_model(df):
    features = ['RSI', 'Vol_10', 'Ret_Lag1', 'Ret_Lag2', 'Vol_Lag1']
    X = df[features].values
    y = df['Target'].values * 100
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)
    return model, features

data = get_data(ticker, horizon=forecast_days)

if data is not None:
    tab_live, tab_audit = st.tabs(["⚡ Live Prediction", "🧪 Custom Range Audit"])

    # --- TAB 1: LIVE ---
    with tab_live:
        model, features = train_model(data)
        last_row = data.iloc[-1]
        vol_limit = data['Vol_10'].quantile(vol_threshold/100)
        is_safe = last_row['Vol_10'] < vol_limit
        pred = model.predict(last_row[features].values.reshape(1, -1))[0]
        
        color = "#00ffcc" if pred > 0.5 and is_safe else "#ff4b4b" if not is_safe else "#ff9500"
        st.markdown(f"<div style='background-color:{color}22; border:2px solid {color}; padding:15px; border-radius:15px; text-align:center;'><h2>{ticker} Projection: {pred:.2f}%</h2></div>", unsafe_allow_html=True)
        
        vdf = data.tail(timeframes[selected_tf])
        y_min, y_max = vdf['Low'].min() * 0.98, vdf['High'].max() * 1.02
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
        fig.add_trace(go.Candlestick(x=vdf.index, open=vdf['Open'], high=vdf['High'], low=vdf['Low'], close=vdf['Close'], name="Price"), row=1, col=1)
        v_colors = ['#00ffcc' if vdf['Close'].iloc[i] >= vdf['Open'].iloc[i] else '#ff4b4b' for i in range(len(vdf))]
        fig.add_trace(go.Bar(x=vdf.index, y=vdf['Volume'], marker_color=v_colors, name="Volume"), row=2, col=1)
        fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False, yaxis1=dict(range=[y_min, y_max]), margin=dict(t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # --- TAB 2: AUDIT + REFINED BREAKDOWN ---
    with tab_audit:
        st.subheader("🕵️ Advanced Walk-Forward Audit")
        c1, c2 = st.columns(2)
        sim_start = c1.date_input("Simulation Start Date", value=datetime.now() - timedelta(days=365))
        sim_end = c2.date_input("Simulation End Date", value=datetime.now())
        
        if st.button("🚀 Run Range Simulation"):
            t_start, t_end = pd.Timestamp(sim_start), pd.Timestamp(sim_end)
            sim_data = data[(data.index >= t_start) & (data.index <= t_end)].copy()
            
            # Tracking Metrics for the Breakdown
            strat_vals, mkt_vals, dates = [10000], [10000], []
            actions = {"BUY": 0, "CASH": 0, "WAIT": 0}
            success_trades = 0
            
            progress = st.progress(0)
            for i in range(0, len(sim_data)-5, 5):
                curr_dt = sim_data.index[i]
                train_set = data[data.index < curr_dt]
                m, fts = train_model(train_set)
                
                row = sim_data.iloc[i]
                dyn_cutoff = train_set['Vol_10'].quantile(vol_threshold/100)
                
                # Logic Execution
                if row['Vol_10'] > dyn_cutoff:
                    realized = 0.0
                    actions["CASH"] += 1
                else:
                    p = m.predict(row[fts].values.reshape(1, -1))[0]
                    if p > 0.5:
                        realized = row['Target'] * 100
                        actions["BUY"] += 1
                        if realized > 0: success_trades += 1
                    else:
                        realized = 0.0
                        actions["WAIT"] += 1

                strat_vals.append(strat_vals[-1] * (1 + realized/100))
                mkt_vals.append(mkt_vals[-1] * (1 + (row['Target'] * 100)/100))
                dates.append(curr_dt)
                progress.progress(i / len(sim_data))
            
            # --- THE REFINED BREAKDOWN ---
            st.divider()
            st.header("📋 Strategy Post-Mortem")
            
            b1, b2, b3 = st.columns(3)
            with b1:
                st.write("### 📥 Data Intake")
                st.info(f"""
                - **Date Range:** {sim_start} to {sim_end}
                - **Total Sample:** {len(sim_data)} Trading Days
                - **Training Points:** Each step re-trained on up to {len(data[data.index < t_start])} prior days.
                - **Feature Set:** RSI, Volatility, and Lagged Returns.
                """)
            
            with b2:
                st.write("### 🧠 Logic Interpretation")
                total_decisions = sum(actions.values())
                st.success(f"""
                - **Regime Defense:** Stayed in Cash {actions['CASH']} times ({ (actions['CASH']/total_decisions)*100:.1f}% of era).
                - **Offensive Bias:** Placed {actions['BUY']} 'Long' bets.
                - **Patience:** Sat Neutral {actions['WAIT']} times.
                """)
            
            with b3:
                st.write("### 📊 Execution Result")
                final_roi = ((strat_vals[-1]/10000)-1)*100
                hit_rate = (success_trades / actions['BUY'] * 100) if actions['BUY'] > 0 else 0
                st.warning(f"""
                - **Net Profit:** {final_roi:.2f}%
                - **Signal Accuracy:** {hit_rate:.1f}% of 'BUY' signals were correct.
                - **Alpha:** {final_roi - (((mkt_vals[-1]/10000)-1)*100):.2f}% vs Market.
                """)

            # Equity Curve Chart
            fig_aud = go.Figure()
            fig_aud.add_trace(go.Scatter(x=dates, y=strat_vals[1:], name="AI Strategy", line=dict(color="#00ffcc", width=3)))
            fig_aud.add_trace(go.Scatter(x=dates, y=mkt_vals[1:], name="Market", line=dict(color="gray", dash='dash')))
            fig_aud.update_layout(template="plotly_dark", title="Backtest Performance", height=400)
            st.plotly_chart(fig_aud, use_container_width=True)
else:
    st.error("Ticker not found.")


