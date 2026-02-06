import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Institutional Earnings Terminal", page_icon="🏦")

# --- Custom Styling ---
st.markdown("""
    <style>
    .main { background-color: #0b0e14; }
    .report-box { 
        background-color: #161b22; 
        border-left: 5px solid #ff9500; 
        padding: 20px; 
        border-radius: 5px; 
        margin-bottom: 25px;
    }
    .status-up { color: #00ffcc; font-weight: bold; }
    .status-down { color: #ff4b4b; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- Logic Modules ---

@st.cache_data(ttl=3600)
def fetch_raw_data(ticker):
    """
    Fetches only serializable data. 
    Returning a Ticker object causes the UnserializableReturnValueError.
    """
    try:
        tk = yf.Ticker(ticker)
        # 1. Fetch Price History
        df = tk.history(period="1y", interval="1d")
        if df.empty:
            return None, None
        
        # Standardize columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 2. Extract Earnings Date as a string/simple date
        earnings_date = None
        try:
            calendar = tk.calendar
            if calendar is not None and not calendar.empty:
                if isinstance(calendar, pd.DataFrame):
                    earnings_date = calendar.iloc[0, 0]
                else:
                    earnings_date = calendar.get('Earnings Date', [None])[0]
        except:
            pass
            
        return df, earnings_date
    except Exception as e:
        return None, None

def get_ai_narrative(poly_coeffs, forecast_days, current_price, end_price, earnings_date):
    """Generates the descriptive analysis of the projection."""
    a, b, c = poly_coeffs
    direction = "BULLISH" if end_price > current_price else "BEARISH"
    change_pct = ((end_price - current_price) / current_price) * 100
    
    momentum = "Accelerating Momentum" if a > 0 else "Trend Exhaustion"

    earnings_warning = ""
    if earnings_date:
        try:
            e_date = pd.to_datetime(earnings_date).replace(tzinfo=None)
            days_to_earnings = (e_date - datetime.now()).days
            if 0 <= days_to_earnings <= 14:
                earnings_warning = f"⚠️ **EARNINGS EVENT ALERT:** Scheduled for approx. {days_to_earnings} days. Expect high aftermarket volatility."
        except:
            pass

    return f"""
    ### 🧠 AI Projection Narrative
    **Horizon:** {forecast_days} Days | **Projected Trend:** <span class="{'status-up' if direction == 'BULLISH' else 'status-down'}">{change_pct:+.2f}%</span>
    **Curve Analysis:** {momentum} detected via quadratic regression.
    
    {earnings_warning}
    
    *Technical Context: This model evaluates daily closes against aftermarket 'gap' patterns to project the dashed trendline.*
    """

# --- Sidebar UI ---
with st.sidebar:
    st.header("🏛️ Terminal Settings")
    ticker_input = st.text_input("Tickers (Comma Separated)", "NVDA, TSLA, AAPL")
    tickers = [t.strip().upper() for t in ticker_input.split(",")]
    
    st.divider()
    forecast_horizon = st.slider("AI Projection Horizon (Days)", 1, 14, 10)
    st.caption("Adjust the horizon to see how the quadratic curve extends into future sessions.")

# --- Main Dashboard Loop ---
st.title("Institutional Intelligence Terminal")
st.subheader("Earnings Proximity & Aftermarket Shift Analysis")

for symbol in tickers:
    # Crucial: fetch_raw_data returns serializable types only
    df, next_earnings = fetch_raw_data(symbol)
    
    if df is not None and len(df) > 30:
        # 1. Aftermarket Shift Logic (Close to next Open)
        df['Aftermarket_Shift'] = (df['Open'].shift(-1) - df['Close']) / df['Close'] * 100
        
        # 2. Prediction Math
        y = df["Close"].values
        x = np.arange(len(y))
        poly_coeffs = np.polyfit(x, y, 2)
        future_x = np.arange(len(y), len(y) + forecast_horizon)
        prediction = np.polyval(poly_coeffs, future_x)
        
        pred_dates = [df.index[-1]] + [df.index[-1] + timedelta(days=i+1) for i in range(forecast_horizon)]
        pred_prices = [df["Close"].iloc[-1]] + list(prediction)

        # --- Dashboard Layout ---
        st.header(f"📈 {symbol} Terminal View")
        
        # Narrative Section
        report = get_ai_narrative(poly_coeffs, forecast_horizon, df["Close"].iloc[-1], pred_prices[-1], next_earnings)
        st.markdown(f'<div class="report-box">{report}</div>', unsafe_allow_html=True)

        # Master Plotly Object
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, 
            vertical_spacing=0.08, row_heights=[0.7, 0.3],
            subplot_titles=("Price Action & AI Projection", "Aftermarket Gap Analysis (%)")
        )

        # Row 1: Candlesticks & AI Dashed Line
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=pred_dates, y=pred_prices, name='AI Forecast', 
            line=dict(color='#00ffcc', width=4, dash='dashdot')
        ), row=1, col=1)

        # Row 2: Aftermarket Gaps (Green for Gap Up, Red for Gap Down)
        colors = ['#ff4b4b' if val < 0 else '#00ffcc' for val in df['Aftermarket_Shift']]
        fig.add_trace(go.Bar(
            x=df.index, y=df['Aftermarket_Shift'], name='Gap %', marker_color=colors
        ), row=2, col=1)

        # UI Refinements
        fig.update_layout(
            template="plotly_dark", height=800, 
            xaxis_rangeslider_visible=False, 
            hovermode="x unified",
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        # Add Timeframe Selectors
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(step="all")
                ]),
                bgcolor="#161b22"
            ),
            row=2, col=1
        )

        st.plotly_chart(fig, use_container_width=True)
        st.divider()
    else:
        st.error(f"Ticker {symbol}: Data inaccessible or insufficient history.")


