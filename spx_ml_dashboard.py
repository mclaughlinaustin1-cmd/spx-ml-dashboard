import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import datetime

st.set_page_config(layout="wide")

# ---------------- SESSION STATE ---------------- #

if "trades" not in st.session_state:
    st.session_state.trades = []

if "cash" not in st.session_state:
    st.session_state.cash = 10000.0

# ---------------- DATA LOADER ---------------- #

@st.cache_data
def load_data(ticker, period):
    df = yf.download(ticker, period=period, progress=False)

    # flatten weird yfinance multi-index
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.dropna()
    return df

# ---------------- INDICATORS ---------------- #

def add_indicators(df):
    df = df.copy()

    close = df["Close"].astype(float)

    delta = close.diff()

    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()

    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    return df

# ---------------- FORECAST ---------------- #

def forecast(df, days=30):
    if len(df) < 20:
        return None, None

    y = df["Close"].values
    X = np.arange(len(y)).reshape(-1,1)

    model = LinearRegression().fit(X, y)

    future_X = np.arange(len(y), len(y)+days).reshape(-1,1)
    preds = model.predict(future_X)

    future_dates = pd.date_range(df.index[-1], periods=days+1)[1:]

    return future_dates, preds

# ---------------- CHART ---------------- #

def plot_chart(df, ticker, chart_type, show_rsi, show_macd, show_forecast):

    fig = go.Figure()

    if chart_type == "Candles":
        fig.add_candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"]
        )
    else:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["Close"],
            name="Price",
            line=dict(width=2)
        ))

    ymin = float(df["Low"].min())
    ymax = float(df["High"].max())

    # Trades overlay
    for t in st.session_state.trades:
        if t["ticker"] == ticker:
            color = "green" if t["type"]=="BUY" else "red"

            fig.add_trace(go.Scatter(
                x=[t["date"]],
                y=[t["price"]],
                mode="markers",
                marker=dict(size=12, color=color),
                name=t["type"]
            ))

            ymin = min(ymin, t["price"])
            ymax = max(ymax, t["price"])

    # Forecast overlay
    if show_forecast:
        fd, fp = forecast(df)
        if fd is not None:
            fig.add_trace(go.Scatter(
                x=fd,
                y=fp,
                name="Forecast",
                line=dict(dash="dot", width=2)
            ))
            ymin = min(ymin, float(fp.min()))
            ymax = max(ymax, float(fp.max()))

    pad = (ymax - ymin) * 0.05
    ymin -= pad
    ymax += pad

    fig.update_layout(
        template="plotly_dark",
        height=550,
        yaxis=dict(range=[ymin, ymax]),
        xaxis_rangeslider_visible=False,
        title=ticker
    )

    st.plotly_chart(fig, use_container_width=True)

    # RSI
    if show_rsi and "RSI" in df:
        st.subheader("RSI")
        st.line_chart(df["RSI"].dropna())

    # MACD
    if show_macd and {"MACD","Signal"}.issubset(df.columns):
        st.subheader("MACD")
        st.line_chart(df[["MACD","Signal"]].dropna())

# ---------------- SIDEBAR ---------------- #

st.sidebar.title("📊 Controls")

ticker = st.sidebar.text_input("Ticker", "AAPL").upper()

period = st.sidebar.selectbox(
    "Range",
    ["1mo","3mo","6mo","1y","3y","5y"]
)

chart_type = st.sidebar.radio("Chart", ["Line","Candles"])

show_rsi = st.sidebar.checkbox("RSI", True)
show_macd = st.sidebar.checkbox("MACD", True)
show_forecast = st.sidebar.checkbox("Forecast", True)

qty = st.sidebar.number_input("Shares", 1, 1000, 1)

# ---------------- MAIN ---------------- #

df = load_data(ticker, period)

if df.empty:
    st.warning("No data found")
    st.stop()

df = add_indicators(df)

current_price = float(df["Close"].iloc[-1])

st.metric("Current Price", f"${current_price:,.2f}")

# ---------------- TRADING ---------------- #

b1, b2 = st.sidebar.columns(2)

if b1.button("📈 BUY"):
    cost = current_price * qty
    if st.session_state.cash >= cost:
        st.session_state.cash -= cost
        st.session_state.trades.append({
            "ticker": ticker,
            "type": "BUY",
            "price": current_price,
            "qty": qty,
            "date": df.index[-1]
        })

if b2.button("📉 SELL"):
    st.session_state.cash += current_price * qty
    st.session_state.trades.append({
        "ticker": ticker,
        "type": "SELL",
        "price": current_price,
        "qty": qty,
        "date": df.index[-1]
    })

# ---------------- CHART ---------------- #

plot_chart(
    df,
    ticker,
    chart_type,
    show_rsi,
    show_macd,
    show_forecast
)

# ---------------- PORTFOLIO ---------------- #

st.subheader("💼 Portfolio")

profit = 0.0

for t in st.session_state.trades:
    if t["ticker"] == ticker:
        if t["type"] == "BUY":
            profit += (current_price - t["price"]) * t["qty"]
        else:
            profit += (t["price"] - current_price) * t["qty"]

st.metric("Cash", f"${st.session_state.cash:,.2f}")
st.metric("P/L", f"${profit:,.2f}")

if st.session_state.trades:
    st.dataframe(pd.DataFrame(st.session_state.trades))
