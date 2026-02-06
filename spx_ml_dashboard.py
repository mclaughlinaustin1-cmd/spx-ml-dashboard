import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

# --- Page Config ---
st.set_page_config(layout="wide", page_title="AI Alpha Terminal v7.1", page_icon="🏛️")

# --- Global Sidebar ---
with st.sidebar:
    st.header("⚙️ Quant Lab")
    ticker = st.text_input("Ticker Symbol", "NVDA").upper()
    forecast_days = st.slider("Forecast Horizon (Days)", 1, 10, 5)
    
    st.divider()
    st.caption("v7.1: Regime-Adaptive Volatility Filtering")
    # NEW CONTROL: The "Safety Switch"
    vol_threshold = st.slider("Volatility Safety Cutoff (%)", 50, 100, 85, 
                              help="If market volatility is in the top X% percentile, the AI will go to Cash (Safety).")

# --- Core Processing Logic ---
@st.cache_data(ttl=3600)
def get_data_with_lags(ticker, horizon=5):
    df = yf.download(ticker, period="10y", interval="1d")
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # 1. Feature Engineering
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
    
    # Reduced depth to prevent overfitting on noise
    model = RandomForestRegressor(n_estimators=150, max_depth=7, min_samples_leaf=10, random_state=42)
    model.fit(X, y)
    return model, features

data = get_data_with_lags(ticker, horizon=forecast_days)

if data is not None:
    tab_live, tab_audit = st.tabs(["⚡ Live Prediction", "🧪 Walk-Forward Lab"])

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
            color = "#ff4b4b" # Red
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

    # --- TAB 2: REGIME-ADAPTIVE WALK-FORWARD ---
    with tab_audit:
        st.subheader("🧪 Regime-Adaptive Walk-Forward")
        st.markdown(f"**The Fix:** This simulation calculates the **{vol_threshold}th Percentile** of volatility. If the market exceeds this, the AI goes to CASH (0% return) to avoid crashes.")
        
        if st.button("🚀 Run Adaptive Simulation"):
            # Set up simulation
            start_date = data.index.max() - timedelta(days=730) # Last 2 years
            sim_data = data[data.index >= start_date].copy()
            
            dates = []
            strat_equity = [10000]
            market_equity = [10000]
            
            # Pre-calculate Volatility Cutoff based on "Past" data only
            # In a real engine, this would update dynamically, but fixed is okay for demo
            initial_train = data[data.index < start_date]
            dynamic_cutoff = initial_train['Vol_10'].quantile(vol_threshold/100)
            
            progress = st.progress(0)
            step_size = 5 # Weekly re-balance
            
            for i in range(0, len(sim_data) - step_size, step_size):
                curr_date = sim_data.index[i]
                
                # 1. Train on PAST
                train_sub = data[data.index < curr_date]
                m, feats = train_model(train_sub)
                
                # 2. Check Regime
                curr_vol = sim_data.iloc[i]['Vol_10']
                
                # 3. Predict
                if curr_vol > dynamic_cutoff:
                    # REGIME FILTER ACTIVATED -> GO TO CASH
                    realized_return = 0.0 
                    action = "CASH"
                else:
                    # REGIME SAFE -> TAKE THE TRADE
                    X_test = sim_data.iloc[i][feats].values.reshape(1, -1)
                    pred = m.predict(X_test)[0]
                    
                    # Actual market move over next step_size days
                    # Note: We use the 'Target' (future return) from the data
                    actual_move = sim_data.iloc[i]['Target'] * 100
                    
                    # If model says UP and pred > 0, we get the return. 
                    # If model says DOWN and pred < 0, we get 0 (or short). 
                    # Let's assume Long-Only for simplicity:
                    if pred > 0:
                        realized_return = actual_move
                        action = "BUY"
                    else:
                        realized_return = 0.0 # Stay in cash
                        action = "WAIT"

                # Update Equity
                new_strat = strat_equity[-1] * (1 + realized_return/100)
                strat_equity.append(new_strat)
                
                # Update Market (Buy & Hold)
                mkt_move = sim_data.iloc[i]['Target'] * 100
                new_mkt = market_equity[-1] * (1 + mkt_move/100)
                market_equity.append(new_mkt)
                
                dates.append(curr_date)
                progress.progress(min((i + step_size) / len(sim_data), 1.0))
            
            # Plotting
            fig_res = go.Figure()
            # Align dates (we have 1 more equity point than dates, so slice)
            plot_dates = dates 
            
            fig_res.add_trace(go.Scatter(x=plot_dates, y=strat_equity[1:], name="AI (Adaptive)", line=dict(color='#00ffcc', width=2)))
            fig_res.add_trace(go.Scatter(x=plot_dates, y=market_equity[1:], name="Buy & Hold", line=dict(color='gray', dash='dash')))
            fig_res.update_layout(template="plotly_dark", title=f"Performance: Adaptive AI vs Market (Vol Cutoff: {vol_threshold}%)")
            st.plotly_chart(fig_res, use_container_width=True)
            
            # Statistics
            strat_ret = (strat_equity[-1] - 10000) / 100
            mkt_ret = (market_equity[-1] - 10000) / 100
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Adaptive Strategy Return", f"{strat_ret:.1f}%")
            c2.metric("Buy & Hold Return", f"{mkt_ret:.1f}%")
            c3.metric("Alpha (Edge)", f"{strat_ret - mkt_ret:.1f}%", delta_color="normal")
            
            if strat_ret > mkt_ret:
                st.success("✅ The Volatility Filter worked! By sitting out the 'crash days', the AI preserved capital and outperformed.")
            else:
                st.info("The market was steady. The filter may have been too sensitive, causing missed opportunities.")
