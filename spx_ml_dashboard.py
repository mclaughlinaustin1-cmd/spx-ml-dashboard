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

st.set_page_config("AI Trading Platform", layout="wide")
st.title("📈 AI Trading Platform with LSTM & Alerts")

# ================= DATA =================
@st.cache_data(ttl=1800)
def load_data(ticker, days):
    end = datetime.today()
    start = end - timedelta(days=days)
    df = yf.download(ticker, start=start, end=end)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

# ================= INDICATORS =================
def add_indicators(df):
    df = df.copy()
    close = df["Close"]
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

# ================= TREND MODEL =================
def trend_probability(df):
    df = df.copy()
    df["Future"] = df["Close"].shift(-3)
    df["Target"] = (df["Future"] > df["Close"]).astype(int)
    df = df.dropna()
    if len(df) < 10:
        return 0.5
    features = ["RSI","MACD","Signal","Volatility"]
    X = df[features].dropna()
    y = df.loc[X.index, "Target"]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X_scaled, y)
    latest = df[features].iloc[-1:].fillna(method="bfill")
    latest_scaled = scaler.transform(latest)
    return model.predict_proba(latest_scaled)[0][1]

# ================= SIGNALS =================
def signal(rsi, macd, sig):
    if rsi < 30 and macd > sig:
        return "BUY"
    if rsi > 70 and macd < sig:
        return "SELL"
    return "HOLD"

def risk_level(vol):
    if vol < 0.01: return "LOW"
    if vol < 0.025: return "MEDIUM"
    return "HIGH"

# ================= BACKTEST =================
def backtest(df):
    cash = 10000
    shares = 0
    for i in range(len(df)):
        s = signal(df["RSI"].iloc[i], df["MACD"].iloc[i], df["Signal"].iloc[i])
        price = df["Close"].iloc[i]
        if s == "BUY" and cash > 0:
            shares = cash / price
            cash = 0
        elif s == "SELL" and shares > 0:
            cash = shares * price
            shares = 0
    return round(cash + shares*df["Close"].iloc[-1],2)

# ================= PAPER TRADING =================
if "balance" not in st.session_state:
    st.session_state.balance = 10000
    st.session_state.shares = {}

def paper_trade(ticker, price, action):
    if ticker not in st.session_state.shares:
        st.session_state.shares[ticker] = 0
    if action=="BUY" and st.session_state.balance > 0:
        qty = st.session_state.balance / price
        st.session_state.shares[ticker] += qty
        st.session_state.balance = 0
    elif action=="SELL" and st.session_state.shares[ticker] > 0:
        st.session_state.balance += st.session_state.shares[ticker]*price
        st.session_state.shares[ticker] = 0

# ================= PORTFOLIO OPTIMIZER =================
def optimize_portfolio(prices):
    returns = prices.pct_change().dropna()
    cov = returns.cov()
    mean = returns.mean()
    n = len(mean)
    def risk(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    cons = {"type":"eq","fun":lambda w: np.sum(w)-1}
    bounds = [(0,1)]*n
    w0 = np.ones(n)/n
    res = minimize(risk, w0, bounds=bounds, constraints=cons)
    return res.x

# ================= NEWS SENTIMENT =================
def news_sentiment(ticker):
    try:
        news = yf.Ticker(ticker).news[:5]
        positive = sum(1 for n in news if any(w in n["title"].lower() for w in ["up","beats","growth","profit","surge"]))
        negative = sum(1 for n in news if any(w in n["title"].lower() for w in ["down","drop","loss","fall","miss"]))
        if positive > negative: return "Positive 📈"
        if negative > positive: return "Negative 📉"
        return "Neutral ⚖"
    except:
        return "Neutral ⚖"

# ================= LSTM FORECAST =================
def lstm_forecast(df, steps=7):
    prices = df["Close"].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices.reshape(-1,1))
    X = []
    y = []
    window = 30
    for i in range(window, len(scaled)-steps):
        X.append(scaled[i-window:i,0])
        y.append(scaled[i:i+steps,0])
    if len(X) == 0:
        return []
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1],1))
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X.shape[1],1)))
    model.add(Dense(steps))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X,y,epochs=20,batch_size=16,verbose=0)
    last_window = scaled[-window:].reshape((1,window,1))
    pred_scaled = model.predict(last_window, verbose=0)[0]
    return scaler.inverse_transform(pred_scaled.reshape(-1,1)).flatten()

# ================= CHART =================
def plot_chart(df, ticker, lstm_pred=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD"))
    fig.add_trace(go.Scatter(x=df.index, y=df["Signal"], name="Signal"))
    if lstm_pred is not None:
        future_dates = [df.index[-1]+pd.Timedelta(days=i+1) for i in range(len(lstm_pred))]
        fig.add_trace(go.Scatter(x=future_dates, y=lstm_pred, name="LSTM Forecast", mode="lines+markers"))
    fig.update_layout(title=f"{ticker} Chart", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# ================= UI =================
with st.sidebar:
    tickers_input = st.text_input("Tickers (comma)", "AAPL,MSFT,GOOG")
    days = st.selectbox("History (days)", [180,365,730,1825], index=1)
    run = st.button("Run Platform")

st.subheader("📊 Market Overview")
portfolio_prices = pd.DataFrame()

# ================= APP =================
if run:
    tickers = [t.strip() for t in tickers_input.upper().split(",")]
    results = []

    for t in tickers:
        df = load_data(t, days)
        if len(df) < 50:
            st.warning(f"{t}: Not enough data")
            continue
        df = add_indicators(df)
        prob_up = trend_probability(df)
        sig = signal(df["RSI"].iloc[-1], df["MACD"].iloc[-1], df["Signal"].iloc[-1])
        risk = risk_level(df["Volatility"].iloc[-1])
        bt_val = backtest(df)
        news = news_sentiment(t)
        lstm_pred = lstm_forecast(df)

        results.append([t, round(prob_up*100,1), sig, risk, bt_val, news])

        # Chart with LSTM forecast
        plot_chart(df, t, lstm_pred)

        portfolio_prices[t] = df["Close"]

        if st.button(f"Paper BUY {t}"):
            paper_trade(t, df["Close"].iloc[-1], "BUY")
        if st.button(f"Paper SELL {t}"):
            paper_trade(t, df["Close"].iloc[-1], "SELL")

        # Alert logic
        if sig=="BUY" and prob_up>0.6:
            st.info(f"🚨 ALERT: {t} strong BUY signal with {round(prob_up*100,1)}% trend probability!")
        if sig=="SELL" and prob_up<0.4:
            st.warning(f"⚠ ALERT: {t} strong SELL signal with {round(prob_up*100,1)}% trend probability!")

    if results:
        st.subheader("💼 AI Portfolio Overview")
        table = pd.DataFrame(results, columns=["Ticker","Trend Up %","Signal","Risk","Backtest $10k","News"])
        st.dataframe(table, use_container_width=True)
        st.write("Cash:", round(st.session_state.balance,2))
        st.write("Holdings:", st.session_state.shares)

        if len(portfolio_prices.columns) > 1:
            weights = optimize_portfolio(portfolio_prices)
            st.subheader("📊 Optimized Portfolio Weights")
            opt = pd.DataFrame({"Ticker":portfolio_prices.columns,"Weight":np.round(weights,3)})
            st.dataframe(opt, use_container_width=True)

        st.success("Platform is running with LSTM forecasts and alerts ✅")
