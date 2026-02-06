import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from datetime import datetime, timedelta

# --- Page Config ---
st.set_page_config(layout="wide", page_title="AI Alpha Terminal v7.0", page_icon="🏛️")

# --- Global Sidebar ---
with st.sidebar:
    st.header("⚙️ Quant Lab")
    ticker = st.text_input("Ticker Symbol", "NVDA").upper()
    forecast_days = st.slider("Forecast Horizon (Days)", 1, 10, 5)
    st.divider()
    st.caption(f"v7.0: Walk-Forward Validation & Lag Engineering")

# --- Core Processing Logic (Fine-Tuned) ---
@st.cache_data(ttl=3600)
def get_data_with_lags(ticker, horizon=5):
    # 1. Fetch Data (10y for robust backtesting)
    df = yf.download(ticker, period="10y", interval="1d")
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # 2. Feature Engineering (The "Memory" Upgrade)
    # Standard Indicators
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_10'] = df['Log_Ret'].rolling(10).std()
    
    # Lagged Features (Giving the model "Memory")
    df['Ret_Lag1'] = df['Log_Ret'].shift(1)
    df['Ret_Lag2'] = df['Log_Ret'].shift(2)
    df['Ret_Lag5'] = df['Log_Ret'].shift(5)
    df['Vol_Lag1'] = df['Vol_10'].shift(1)
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    # 3. The Target: Future Return 'horizon' days out (Smoothing noise)
    # We want to predict: (Price_in_5_days / Price_Today) - 1
    df['Target'] = df['Close'].shift(-horizon) / df['Close'] - 1
    
    # Drop NaNs created by lags and target shifting
    df = df.dropna()
    
    return df

def train_model(df):
    features = ['RSI', 'Vol_10', 'Ret_Lag1', 'Ret_Lag2', 'Ret_Lag5', 'Vol_Lag1']
    X = df[features].values
    y = df['Target'].values * 100  # Scale to percentage
    
    # Fine-Tuning: Use more estimators and deeper trees for complex feature interactions
    model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=4, random_state=42)
    model.fit(X, y)
    
    return model, features

data = get_data_with_lags(ticker, horizon=forecast_days)

if data is not None:
    tab_live, tab_audit = st.tabs(["⚡ Live Prediction", "🧪 Walk-Forward Lab"])

    # --- TAB 1: LIVE TERMINAL ---
    with tab_live:
        model, feature_names = train_model(data)
        
        # Latest Data for Prediction
        last_row = data.iloc[-1]
        input_data = last_row[feature_names].values.reshape(1, -1)
        pred_return = model.predict(input_data)[0]
        
        # Calculate Target Price
        current_price = last_row['Close']
        target_price = current_price * (1 + pred_return/100)
        
        # Dynamic Signal
        signal = "BULLISH" if pred_return > 1.0 else "BEARISH" if pred_return < -1.0 else "NEUTRAL"
        sig_color = "#00ffcc" if signal == "BULLISH" else "#ff4b4b" if signal == "BEARISH" else "#ff9500"
        
        st.markdown(f"""
        <div style='background-color:{sig_color}22; border:2px solid {sig_color}; padding:20px; border-radius:10px; text-align:center'>
            <h2 style='margin:0; color:{sig_color}'>{signal} SIGNAL ({forecast_days}-Day Horizon)</h2>
            <p style='margin:0'>AI predicts a <b>{pred_return:.2f}%</b> move to <b>${target_price:,.2f}</b></p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([3, 1])
        with col1:
            # Independent Price Chart
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Price'))
            fig.update_layout(template="plotly_dark", title="Price Action", height=450, xaxis_range=[data.index[-90], data.index[-1]])
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.write("### 🧠 Feature Importance")
            # Show which inputs the AI cares about most
            importances = pd.DataFrame({'Feature': feature_names, 'Weight': model.feature_importances_})
            importances = importances.sort_values(by='Weight', ascending=True)
            
            fig_imp = go.Figure(go.Bar(
                x=importances['Weight'],
                y=importances['Feature'],
                orientation='h',
                marker_color=sig_color
            ))
            fig_imp.update_layout(template="plotly_dark", height=300, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_imp, use_container_width=True)

    # --- TAB 2: WALK-FORWARD AUDIT (REALITY CHECK) ---
    with tab_audit:
        st.subheader("🧪 Walk-Forward Validation")
        st.markdown("This test simulates **re-training the model every month** over the past period. It is the most realistic way to measure 'Future Prediction' capabilities.")
        
        c1, c2 = st.columns(2)
        start_date = c1.date_input("Audit Start", value=data.index.max() - timedelta(days=365))
        
        if st.button("🚀 Run Simulation"):
            # Prepare Simulation Data
            sim_data = data[data.index >= pd.Timestamp(start_date)].copy()
            train_window_start = data.index[0]
            
            predictions = []
            dates = []
            
            progress = st.progress(0)
            
            # THE WALK-FORWARD LOOP
            # We step through the data week by week (every 5 trading days)
            step_size = 5 
            total_steps = len(sim_data) // step_size
            
            for i in range(0, len(sim_data) - step_size, step_size):
                # 1. Define current "Now"
                current_date = sim_data.index[i]
                
                # 2. Train on EVERYTHING strictly before "Now"
                # This ensures zero look-ahead bias
                train_subset = data[data.index < current_date]
                
                if len(train_subset) > 200: # Need min data to train
                    # Train dynamic model
                    m, feats = train_model(train_subset)
                    
                    # 3. Predict for the specific test week
                    # We are predicting the *next* 5 days of returns
                    X_test = sim_data.iloc[i][feats].values.reshape(1, -1)
                    pred = m.predict(X_test)[0]
                    
                    predictions.append(pred)
                    dates.append(current_date)
                
                progress.progress(min((i + step_size) / len(sim_data), 1.0))
            
            # Analysis
            res_df = pd.DataFrame({'Date': dates, 'Predicted_Ret': predictions})
            res_df.set_index('Date', inplace=True)
            
            # Merge with actual 5-day returns to compare
            # Note: The 'Target' column in sim_data is already the REAL future return
            res_df['Actual_Ret'] = sim_data.loc[res_df.index]['Target'] * 100
            
            # Strategy: If Pred > 0, Buy. Else, Hold Cash (0 return)
            res_df['Strategy_Daily'] = np.where(res_df['Predicted_Ret'] > 0, res_df['Actual_Ret'], 0)
            
            # Cumulative Growth
            # We divide by 100 because returns are in %
            res_df['Equity'] = (1 + res_df['Strategy_Daily']/100).cumprod() * 10000
            res_df['Market'] = (1 + res_df['Actual_Ret']/100).cumprod() * 10000
            
            # Visualization
            fig_res = go.Figure()
            fig_res.add_trace(go.Scatter(x=res_df.index, y=res_df['Equity'], name="AI Strategy", line=dict(color='#00ffcc', width=2)))
            fig_res.add_trace(go.Scatter(x=res_df.index, y=res_df['Market'], name="Buy & Hold", line=dict(color='gray', dash='dash')))
            
            st.plotly_chart(fig_res, use_container_width=True)
            
            final_ret = res_df['Equity'].iloc[-1]
            st.metric("Final Account Value ($10k Start)", f"${final_ret:,.2f}")
            
            if final_ret > res_df['Market'].iloc[-1]:
                st.success("✅ The Fine-Tuned Model outperformed the market in this simulation.")
            else:
                st.warning("⚠️ The Model underperformed. The market regime may be too volatile for these features.")
                
else:
    st.error("Invalid Ticker or insufficient data.")
