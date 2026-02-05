import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize
import plotly.graph_objects as go

st.set_page_config("AI Trading Platform", layout="wide")

# ===================== DATA =====================

@st.cache_data(ttl=1800)
def load_data(ticker, days):
    end = datetime.today()
    start = end - timedelta(days=days)
    df = yf.download(ticker, start=start, end=end)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df.dropna()

# ===================== INDICATORS =====================

def indicators(df):
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

# ===================== AI TREND MODEL =====================

def trend_probability(df):
    df["Future"] = df["Close"].shift(-3)
    df["Target"] = (df["Future"] > df["Close"]).astype(int)
    df = df.dropna()

    features = ["RSI","MACD","Signal","Volatility"]
    X = df[features]
    y = df["Target"]

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=300)
    model.fit(X,y)

    latest = scaler.transform(df[features].iloc[-1:].values)
    return model.predict_proba(latest)[0][1]

# ===================== SIGNALS =====================

def signal(rsi, macd, sig):
    if rsi < 30 and macd > sig:
        return "BUY"
    if rsi > 70 and macd < sig:
        return "SELL"
    return "HOLD"

# ===================== BACKTEST =====================

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

    final = cash + shares * df["Close"].iloc[-1]
    return round(final,2)

# ===================== PAPER TRADING =====================

if "balance" not in st.session_state:
    st.session_state.balance = 10000
    st.session_state.shares = {}

def paper_trade(ticker, price, action):
    if ticker not in st.session_state.shares:
        st.session_state.shares[ticker] = 0

    if action == "BUY":
        qty = st.session_state.balance / price
        st.session_state.shares[ticker] += qty
        st.session_state.balance = 0

    elif action == "SELL":
        st.session_state.balance += st.session_state.shares[ticker] * price
        st.session_state.shares[ticker] = 0

# ===================== PORTFOLIO OPTIMIZER =====================

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

# ===================== CHART =====================

def chart(df, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD"))
    fig.add_trace(go.Scatter(x=df.index, y=df["Signal"], name="Signal"))
    fig.update_layout(title=f"{ticker} Chart", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# ===================== UI =====================

st.title("📈 AI Trading Platform")

colA, colB = st.columns([2,1])

with colB:
    tickers = st.text_input("Tickers (comma)", "AAPL,MSFT,GOOG")
    days = st.selectbox("History", [180,365,730,1825], index=1)
    run = st.button("Run Platform")

with colA:
    st.subheader("Market Dashboard")

# ===================== APP =====================

if run:
    tickers = [t.strip() for t in tickers.split(",")]
    portfolio_prices = pd.DataFrame()

    for t in tickers:
        df = indicators(load_data(t, days))

        prob = trend_probability(df)
        sig = signal(df["RSI"].iloc[-1], df["MACD"].iloc[-1], df["Signal"].iloc[-1])
        backtest_value = backtest(df)

        st.markdown(f"### {t}")
        st.write(f"Trend Up Probability: {round(prob*100,1)}%")
        st.write(f"Signal: {sig}")
        st.write(f"Backtest $10k → ${backtest_value}")

        chart(df, t)

        portfolio_prices[t] = df["Close"]

        if st.button(f"Paper BUY {t}"):
            paper_trade(t, df["Close"].iloc[-1], "BUY")
        if st.button(f"Paper SELL {t}"):
            paper_trade(t, df["Close"].iloc[-1], "SELL")

    st.subheader("💰 Paper Trading Portfolio")
    st.write("Cash:", round(st.session_state.balance,2))
    st.write("Holdings:", st.session_state.shares)

    if len(portfolio_prices.columns) > 1:
        weights = optimize_portfolio(portfolio_prices)
        st.subheader("📊 Optimized Portfolio Weights")

        opt = pd.DataFrame({
            "Ticker": portfolio_prices.columns,
            "Weight": np.round(weights,3)
        })

        st.dataframe(opt, use_container_width=True)

    st.success("Platform running!")
