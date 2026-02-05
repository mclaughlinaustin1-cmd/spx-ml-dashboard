import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import requests

st.set_page_config("AI Trading Dashboard", layout="wide")
st.title("📈 AI Trading Platform - TradingView Style")

# ===================== DATA LOADER =====================
@st.cache_data(ttl=1800)
def load_data(ticker, days):
    end = datetime.today()
    start = end - timedelta(days=days)
    interval = "1h" if days <= 30 else "1d"
    df = yf.download(ticker, start=start, end=end, interval=interval)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

# ===================== INDICATORS =====================
def add_indicators(df):
    df = df.copy()
    close = df["Close"]
    if len(close) < 14:
        df["RSI"] = np.nan
        df["MACD"] = np.nan
        df["Signal"] = np.nan
        df["Returns"] = np.nan
        df["Volatility"] = np.nan
        return df
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    df["Returns"] = close.pct_change()
    df["Volatility"] = df["Returns"].rolling(20).std()
    return df.dropna()

# ===================== TREND MODEL =====================
def trend_probability(df):
    df = df.copy()
    if len(df) < 10:
        return 0.5
    df["Future"] = df["Close"].shift(-3)
    df["Target"] = (df["Future"] > df["Close"]).astype(int)
    df = df.dropna()
    features = ["RSI","MACD","Signal","Volatility"]
    X = df[features].dropna()
    y = df.loc[X.index,"Target"]
    if len(X) < 1: return 0.5
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_scaled, y)
    latest = df[features].iloc[-1:].fillna(method="bfill")
    latest_scaled = scaler.transform(latest)
    return model.predict_proba(latest_scaled)[0][1]

# ===================== SIGNALS & RISK =====================
def signal(rsi, macd, sig):
    if pd.isna(rsi) or pd.isna(macd) or pd.isna(sig):
        return "N/A"
    if rsi < 30 and macd > sig: return "BUY"
    if rsi > 70 and macd < sig: return "SELL"
    return "HOLD"

def risk_level(vol):
    if pd.isna(vol): return "N/A"
    if vol < 0.01: return "LOW"
    if vol < 0.025: return "MEDIUM"
    return "HIGH"

# ===================== BACKTEST =====================
def backtest(df):
    if len(df) < 50: return "N/A"
    cash = 10000
    shares = 0
    for i in range(len(df)):
        s = signal(df["RSI"].iloc[i], df["MACD"].iloc[i], df["Signal"].iloc[i])
        price = df["Close"].iloc[i]
        if s=="BUY" and cash>0:
            shares = cash/price
            cash = 0
        elif s=="SELL" and shares>0:
            cash = shares*price
            shares = 0
    return round(cash + shares*df["Close"].iloc[-1],2)

# ===================== PAPER TRADING =====================
if "balance" not in st.session_state:
    st.session_state.balance = 10000
    st.session_state.shares = {}

def paper_trade(ticker, price, action):
    if ticker not in st.session_state.shares:
        st.session_state.shares[ticker] = 0
    if action=="BUY" and st.session_state.balance>0:
        qty = st.session_state.balance/price
        st.session_state.shares[ticker] += qty
        st.session_state.balance = 0
    elif action=="SELL" and st.session_state.shares[ticker]>0:
        st.session_state.balance += st.session_state.shares[ticker]*price
        st.session_state.shares[ticker] = 0

# ===================== PORTFOLIO OPTIMIZER =====================
def optimize_portfolio(prices):
    returns = prices.pct_change().dropna()
    if returns.empty: return [1/len(prices.columns)]*len(prices.columns)
    cov = returns.cov()
    n = len(returns.columns)
    def risk(weights): return np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    cons = {"type":"eq","fun":lambda w: np.sum(w)-1}
    bounds = [(0,1)]*n
    w0 = np.ones(n)/n
    res = minimize(risk, w0, bounds=bounds, constraints=cons)
    return res.x

# ===================== LSTM FORECAST =====================
@st.cache_data(ttl=3600, show_spinner=False)
def lstm_forecast_cached(prices, steps=7):
    if len(prices) < 50: return []
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices.reshape(-1,1))
    X, y = [], []
    window = 30
    for i in range(window, len(scaled)-steps):
        X.append(scaled[i-window:i,0])
        y.append(scaled[i:i+steps,0])
    if len(X) == 0: return []
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0],X.shape[1],1))
    model = Sequential()
    model.add(LSTM(30, activation='relu', input_shape=(X.shape[1],1)))
    model.add(Dense(steps))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X,y,epochs=5,batch_size=8,verbose=0)
    last_window = scaled[-window:].reshape((1,window,1))
    pred_scaled = model.predict(last_window, verbose=0)[0]
    return scaler.inverse_transform(pred_scaled.reshape(-1,1)).flatten()

# ===================== UNUSUAL WHALES PLACEHOLDER =====================
def get_unusual_whales(ticker):
    # Replace with your API key and endpoint
    return [{"type":"Block Trade","size":"$2M","info":"Large option sweep"}]

# ===================== PLOTLY CHART =====================
def plot_chart(df, ticker, lstm_pred=None, zoom=False, show_rsi=True, show_macd=True,
               show_signals=True, show_forecast=True, key=None, x_range=None, y_range=None):
    fig = go.Figure()
    df_plot = df.iloc[-50:] if zoom else df
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["Close"], name="Price"))
    if show_macd and "MACD" in df_plot:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["MACD"], name="MACD"))
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["Signal"], name="Signal"))
    if show_rsi and "RSI" in df_plot:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["RSI"], name="RSI"))
    if show_signals:
        buy_idx = df_plot[(df_plot["RSI"]<30) & (df_plot["MACD"]>df_plot["Signal"])].index
        sell_idx = df_plot[(df_plot["RSI"]>70) & (df_plot["MACD"]<df_plot["Signal"])].index
        fig.add_trace(go.Scatter(x=buy_idx, y=df_plot.loc[buy_idx,"Close"], mode="markers",
                                 marker=dict(size=10,color="green"), name="BUY"))
        fig.add_trace(go.Scatter(x=sell_idx, y=df_plot.loc[sell_idx,"Close"], mode="markers",
                                 marker=dict(size=10,color="red"), name="SELL"))
    if show_forecast and lstm_pred:
        future_dates = [df_plot.index[-1]+pd.Timedelta(days=i+1) for i in range(len(lstm_pred))]
        fig.add_trace(go.Scatter(x=future_dates, y=lstm_pred, mode="lines+markers", name="LSTM Forecast"))

    # Range selectors
    fig.update_layout(
        title=f"{ticker} Price & Indicators",
        hovermode="x unified",
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date",
            range=x_range
        ),
        yaxis=dict(fixedrange=False, range=y_range)
    )
    st.plotly_chart(fig, use_container_width=True, key=key)

# ===================== UI =====================
with st.sidebar:
    tickers_input = st.text_input("Tickers (comma)", "AAPL,MSFT,GOOG")
    range_options = {
        "24 hours":1,
        "1 week":7,
        "1 month":30,
        "6 months":182,
        "1 year":365,
        "3 years":1095,
        "5 years":1825
    }
    selected_range = st.selectbox("Historical range:", options=list(range_options.keys()))
    days = range_options[selected_range]
    show_rsi = st.checkbox("Show RSI", value=True)
    show_macd = st.checkbox("Show MACD", value=True)
    show_signals = st.checkbox("Show Signals", value=True)
    show_forecast = st.checkbox("Show Forecast", value=True)
    run = st.button("Run Platform")

# ===================== INDICATOR EXPLANATION =====================
st.sidebar.subheader("ℹ️ Indicator Explanation")
st.sidebar.markdown("""
- **Price:** Current stock price (Close)
- **RSI:** Relative Strength Index (0–100). <30 oversold (BUY), >70 overbought (SELL)
- **MACD:** Trend indicator (EMA12 - EMA26). MACD>Signal bullish
- **Signal:** EMA9 of MACD, confirms trend changes
- **Trend Probability:** AI chance stock will rise (>60% = likely UP)
- **Volatility:** Price fluctuations (LOW / MEDIUM / HIGH risk)
- **Buy/Sell Signals:** Generated by RSI + MACD
- **LSTM Forecast:** AI future price projection (next 7 periods)
- **Whale Activity:** Major trades detected (Unusual Whales API)
""")

# ===================== APP =====================
st.subheader("📊 Market Overview")
portfolio_prices = pd.DataFrame()

if run:
    tickers = [t.strip().upper() for t in tickers_input.split(",")]
    results = []

    for t in tickers:
        df = load_data(t, days)
        if df.empty:
            st.warning(f"{t}: No data available for selected range")
            continue
        df = add_indicators(df)

        # Safe signal & probability
        if df.empty or "RSI" not in df or df["RSI"].isna().all():
            sig = "N/A"
            prob_up = 0.5
            risk = "N/A"
            bt_val = "N/A"
            lstm_pred = []
        else:
            prob_up = trend_probability(df)
            sig = signal(df["RSI"].iloc[-1], df["MACD"].iloc[-1], df["Signal"].iloc[-1])
            risk = risk_level(df["Volatility"].iloc[-1])
            bt_val = backtest(df)
            lstm_pred = lstm_forecast_cached(df["Close"].values)

        whales = get_unusual_whales(t)
        news = "Placeholder News"

        results.append([t, round(prob_up*100,1), sig, risk, bt_val, whales, news])

        # Interactive chart with adjustable ranges
        x_range = st.sidebar.date_input(f"{t} X-axis range", [df.index.min().date(), df.index.max().date()])
        y_range = st.sidebar.slider(f"{t} Y-axis range", float(df["Close"].min()), float(df["Close"].max()),
                                    (float(df["Close"].min()), float(df["Close"].max())))

        tab1, tab2 = st.tabs([f"{t} Full Chart", f"{t} Forecast Zoom"])
        with tab1:
            plot_chart(df, t, lstm_pred, zoom=False, show_rsi=show_rsi, show_macd=show_macd,
                       show_signals=show_signals, show_forecast=show_forecast,
                       key=f"{t}_full", x_range=x_range, y_range=y_range)
        with tab2:
            plot_chart(df, t, lstm_pred, zoom=True, show_rsi=show_rsi, show_macd=show_macd,
                       show_signals=show_signals, show_forecast=show_forecast,
                       key=f"{t}_zoom", x_range=x_range, y_range=y_range)

        portfolio_prices[t] = df["Close"]

        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"Paper BUY {t}"):
                paper_trade(t, df["Close"].iloc[-1], "BUY")
        with col2:
            if st.button(f"Paper SELL {t}"):
                paper_trade(t, df["Close"].iloc[-1], "SELL")

        if sig=="BUY" and prob_up>0.6:
            st.info(f"🚨 ALERT: {t} strong BUY signal ({round(prob_up*100,1)}%)")
        if sig=="SELL" and prob_up<0.4:
            st.warning(f"⚠ ALERT: {t} strong SELL signal ({round(prob_up*100,1)}%)")

    if results:
        st.subheader("💼 AI Portfolio Overview")
        table = pd.DataFrame(results, columns=["Ticker","Trend Up %","Signal","Risk","Backtest $10k","Whales Activity","News"])
        st.dataframe(table, use_container_width=True)
        st.write("Cash:", round(st.session_state.balance,2))
        st.write("Holdings:", st.session_state.shares)

        if len(portfolio_prices.columns)>1:
            weights = optimize_portfolio(portfolio_prices)
            st.subheader("📊 Optimized Portfolio Weights")
            opt = pd.DataFrame({"Ticker":portfolio_prices.columns,"Weight":np.round(weights,3)})
            st.dataframe(opt, use_container_width=True)
