import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

# --- Page Config ---
st.set_page_config(layout="wide", page_title="AI Alpha Terminal v7.2", page_icon="🏛️")

# --- Global Sidebar ---
with st.sidebar:
    st.header("⚙️ Quant Lab")
    ticker = st.text_input("Ticker Symbol", "NVDA").upper()
    forecast_days = st.slider("Forecast Horizon (Days)", 1, 10, 5)
    
    st.divider()
    st.caption("v7.2: Custom Date Range Walk-Forward")
    # THE SAFETY SWITCH
    vol_threshold = st.slider("Volatility Safety Cutoff (%)", 50, 100, 85, 
                              help="If market volatility is in the top X% percentile, the AI will go to Cash.")

# --- Core Processing Logic ---
@st.cache_data(ttl=3600)
def get_data_with_lags(ticker, horizon=5):
    # Fetch 10y to ensure we have enough "Pre-Start" data for training
    df = yf.download(ticker, period="10y", interval="1d")
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Feature Engineering
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_10'] = df['Log_Ret'].rolling(10).std()
    
    # Lags (Memory)
    df['Ret_Lag1'] = df['Log_Ret'].shift(1)
    df['Ret_Lag2'] = df['Log_Ret'].shift(2)
    df['Ret_Lag5'] = df['Log_Ret'].shift(5)
    df['Vol_Lag1'] = df['Vol_10'].shift(1)
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    # Target: Return 'horizon' days out
    df['Target'] = df['Close'].shift(-horizon) / df['Close'] - 1
    
    return df.dropna()

def train_model(df):
    features = ['RSI', 'Vol_10', 'Ret_Lag1', 'Ret_Lag2', 'Ret_Lag5', 'Vol_Lag1']
    X = df[features].values
    y = df['Target'].values * 100
    
    model = RandomForestRegressor(n_estimators=150, max_depth=7, min_samples_leaf=10, random_state=42)
    model.fit(X, y)
    return model, features

data = get_data_with_lags(ticker, horizon=forecast_days)

if data is not None:
    tab_live, tab_audit = st.tabs(["⚡ Live Prediction", "🧪 Range-Based Walk-Forward"])

    # --- TAB 1: LIVE TERMINAL ---
    with tab_live:
        model, feature_names = train_model(data)
        last_row = data.iloc[-1]
        
        # Regime Check
        current_vol = last_row['Vol_10']
        vol_cutoff = data['Vol_10'].quantile(vol_threshold/100)
        is_safe = current_vol < vol_cutoff
        
        input_data = last_row[feature_names].values.reshape(1, -1)
        pred_return = model.predict(input_data)[0]
        
        # Display Logic
        if not is_safe:
            status = "CASH (High Volatility)"
            color = "#ff4b4b"
            advice = "Market is too erratic. Strategy is in safety mode."
        else:
            status = "BULLISH" if pred_return > 0.5 else "BEARISH" if pred_return < -0.5 else "NEUTRAL"
            color = "#00ffcc" if status == "BULLISH" else "#ff9500"
            advice = f"Market is stable. AI predicts {pred_return:.2f}% move."

        st.markdown(f"""
        <div style='background-color:{color}22; border:2px solid {color}; padding:20px; border-radius:10px; text-align:center'>
            <h2 style='margin:0; color:{color}'>{status}</h2>
            <p style='margin:0'>{advice}</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([3, 1])
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close']))
            fig.update_layout(template="plotly_dark", title="Price Action", height=450, xaxis_range=[data.index[-90], data.index[-1]])
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.metric("Current Volatility", f"{current_vol*100:.2f}%")
            st.metric("Safety Cutoff", f"{vol_cutoff*100:.2f}%")
            st.progress(min(current_vol/vol_cutoff, 1.0))
            st.caption("Bar Full = DANGER ZONE")

    # --- TAB 2: RANGE-BASED WALK-FORWARD ---
    with tab_audit:
        st.subheader("🧪 Targeted Era Stress Test")
        st.markdown(f"**Instructions:** Select a Start and End date. The AI will simulate trading through that specific window, re-training itself every week.")
        
        # DATE RANGE SELECTORS
        min_date = data.index.min().date()
        max_date = data.index.max().date()
        
        c_sel1, c_sel2 = st.columns(2)
        
        # Start Date (Must have at least 1 year of data before it for training)
        audit_start = c_sel1.date_input("Simulation Start Date", 
                                       value=max_date - timedelta(days=365),
                                       min_value=min_date + timedelta(days=365),
                                       max_value=max_date - timedelta(days=30))
        
        # End Date
        audit_end = c_sel2.date_input("Simulation End Date", 
                                      value=max_date,
                                      min_value=audit_start + timedelta(days=7),
                                      max_value=max_date)

        if st.button("🚀 Run Range Simulation"):
            # Filter Data to the specific window
            t_start = pd.Timestamp(audit_start)
            t_end = pd.Timestamp(audit_end)
            
            # The 'sim_data' is the future path we are walking through
            sim_data = data[(data.index >= t_start) & (data.index <= t_end)].copy()
            
            if len(sim_data) > 10:
                dates = []
                strat_equity = [10000]
                market_equity = [10000]
                
                # Dynamic Volatility Cutoff Setup
                # We calculate the cutoff based on data BEFORE the simulation starts
                pre_sim_data = data[data.index < t_start]
                dynamic_cutoff = pre_sim_data['Vol_10'].quantile(vol_threshold/100)
                
                progress = st.progress(0)
                step_size = 5 # Weekly re-balance
                
                # WALK-FORWARD LOOP
                for i in range(0, len(sim_data) - step_size, step_size):
                    curr_date = sim_data.index[i]
                    
                    # 1. Train on PAST (Everything before current simulation step)
                    train_sub = data[data.index < curr_date]
                    m, feats = train_model(train_sub)
                    
                    # 2. Check Regime (Volatility Safety)
                    curr_vol = sim_data.iloc[i]['Vol_10']
                    
                    # 3. Predict or Defend
                    if curr_vol > dynamic_cutoff:
                        # Safety Mode
                        realized_return = 0.0 
                        action = "CASH"
                    else:
                        # Active Mode
                        X_test = sim_data.iloc[i][feats].values.reshape(1, -1)
                        pred = m.predict(X_test)[0]
                        
                        actual_move = sim_data.iloc[i]['Target'] * 100
                        
                        # Strategy: Long Only (Buy if > 0)
                        if pred > 0:
                            realized_return = actual_move
                            action = "BUY"
                        else:
                            realized_return = 0.0
                            action = "WAIT"

                    # Update Account
                    new_strat = strat_equity[-1] * (1 + realized_return/100)
                    strat_equity.append(new_strat)
                    
                    mkt_move = sim_data.iloc[i]['Target'] * 100
                    new_mkt = market_equity[-1] * (1 + mkt_move/100)
                    market_equity.append(new_mkt)
                    
                    dates.append(curr_date)
                    progress.progress(min((i + step_size) / len(sim_data), 1.0))
                
                # Results Visualization
                st.divider()
                
                # Calculate final stats
                final_strat = strat_equity[-1]
                final_mkt = market_equity[-1]
                strat_ret = (final_strat - 10000) / 100
                mkt_ret = (final_mkt - 10000) / 100
                
                c_res1, c_res2, c_res3 = st.columns(3)
                c_res1.metric("Strategy Return", f"{strat_ret:.1f}%", f"${final_strat:,.0f}")
                c_res2.metric("Market Return", f"{mkt_ret:.1f}%", f"${final_mkt:,.0f}")
                c_res3.metric("Edge (Alpha)", f"{strat_ret - mkt_ret:.1f}%", delta_color="normal")

                fig_res = go.Figure()
                plot_dates = dates
                fig_res.add_trace(go.Scatter(x=plot_dates, y=strat_equity[1:], name="AI (Adaptive)", line=dict(color='#00ffcc', width=2)))
                fig_res.add_trace(go.Scatter(x=plot_dates, y=market_equity[1:], name="Buy & Hold", line=dict(color='gray', dash='dash')))
                fig_res.update_layout(template="plotly_dark", title=f"Performance: {audit_start} to {audit_end}", height=500)
                st.plotly_chart(fig_res, use_container_width=True)
                
                if strat_ret > mkt_ret:
                    st.success(f"✅ The Strategy beat the market by {strat_ret - mkt_ret:.1f}% during this period.")
                else:
                    st.warning(f"⚠️ The Strategy trailed the market. The volatility filter may have been too aggressive.")
            else:
                st.error("Selected range is too short. Please select at least 2 weeks.")
else:
    st.error("Invalid Ticker or insufficient data.")

