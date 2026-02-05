import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

st.set_page_config(page_title="AI Trading Dashboard", layout="wide")
st.title("📊 AI Trading Intelligence Platform")

# ================= DATA =================

@st.cache_data(ttl=1800)
def fetch_data(ticker, days):
    end = datetime.today()
    start = end - timedelta(days=days)
    df = yf.download(ticker, start=start, end=end)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df.dropna()

# ================= INDICATORS =================

def add_indicators(df):
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

    df = df.dropna()
    return df

# ================= ML TREND MODEL =================

def train_trend_model(df):
    df["Future"] = df["Close"].shift(-3)
    df["Target"] = (df["Future"] > df["Close"]).astype(int)
    df = df.dropna()

    features = ["RSI", "MACD", "Signal", "Volatility"]
    X = df[features]
    y = df["Target"]

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X, y)

    latest = scaler.transform(df[features].iloc[-1:].values)
    prob_up = model.predict_proba(latest)[0][1]

    return prob_up

# ================= SIGNAL LOGIC =================

def trading_signal(rsi, macd, signal):
    if rsi < 30 and macd > signal:
        return "🟢 BUY"
    elif rsi > 70 and macd < signal:
        return "🔴 SELL"
    else:
        return "🟡 HOLD"

def risk_level(vol):
    if vol < 0.01:
        return "LOW"
    elif vol < 0.025:
        return "MEDIUM"
    else:
        return "HIGH"

# ================= NEWS SENTIMENT =================

def news_sentiment(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news[:5]

        positive = 0
        negative = 0

        for n in news:
            title = n["title"].lower()
            if any(w in title for w in ["up", "beats", "growth", "surge", "profit"]):
                positive += 1
            if any(w in title for w in ["down", "miss", "drop", "loss", "fall"]):
                negative += 1

        if positive > negative:
            return "📈 Positive"
        elif negative > positive:
            return "📉 Negative"
        else:
            return "⚖ Neutral"

    except:
        return "⚖ Neutral"

# ================= PLOT =================

def plot_chart(df, ticker):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD"))
    fig.add_trace(go.Scatter(x=df.index, y=df["Signal"], name="Signal"))

    fig.update_layout(
        title=f"{ticker} Price & Indicators",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

# ================= UI =================

with st.sidebar:
    tickers = st.text_input("Stock tickers (comma separated)", "AAPL,MSFT").upper()
    days = st.selectbox("History range", [180, 365, 730, 1825], index=1)
    run = st.button("Run Analysis")

# ================= APP =================

if run:
    stocks = [t.strip() for t in tickers.split(",")]

    results = []

    for ticker in stocks:
        df = fetch_data(ticker, days)

        if df.empty or len(df) < 80:
            st.warning(f"{ticker}: not enough data")
            continue

        df = add_indicators(df)

        prob_up = train_trend_model(df)

        rsi = df["RSI"].iloc[-1]
        macd = df["MACD"].iloc[-1]
        signal_line = df["Signal"].iloc[-1]
        vol = df["Volatility"].iloc[-1]

        signal = trading_signal(rsi, macd, signal_line)
        risk = risk_level(vol)
        sentiment = news_sentiment(ticker)

        results.append([
            ticker,
            round(prob_up * 100, 1),
            signal,
            risk,
            sentiment
        ])

        st.subheader(f"{ticker} Chart")
        plot_chart(df, ticker)

    if results:
        st.subheader("📊 AI Market Overview")

        table = pd.DataFrame(
            results,
            columns=["Ticker", "Trend Up %", "Signal", "Risk", "News Sentiment"]
        )

        st.dataframe(table, use_container_width=True)

        st.success("Analysis complete!")
