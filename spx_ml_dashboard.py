import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import datetime

st.set_page_config(layout="wide")

# ---------------- STATE ---------------- #

if "trades" not in st.session_state:
    st.session_state.trades = []

if "cash" not in st.session_state:
    st.session_state.cash = 10000.0

# ---------------- DATA ---------------- #

@st.cache_data
def load_data(ticker, period):
    df = yf.download(ticker, period=period, interval="1d", progress=False)
    df = df.dropna()
    return df

# ---------------- INDICATORS ---------------- #

def add_indicators(df):
    delta = df["Close"].diff()

    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()

    rs = gain / loss
    df["RSI"] = 100 - 100 / (1 + rs)

    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()

    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9).mean()

    return df

# ---------------- FORECAST ---------------- #

def linear_forecast(df, days=30):
    if len(df) < 20:
        return None, None

    X = np.arange(len(df)).reshape(-1,1)
    y = df["Close"].values.astype(float)

    model = LinearRegression().fit(X, y)

    future_X = np.arange(len(df), len(df)+days).reshape(-1,1)
    preds = model.predict(future_X)

    future_dates = pd.date_range(df.index[-1], periods=days+1)[1:]

    return future_dates, preds.astype(float)

# ---------------- CHART ---------------- #

def plot_chart(df, ticker, chart_type, show_rsi, show_macd, show_forecast):

    fig = go.Figure()

    if chart_type == "Candles":
        fig.add_candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price"
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

    # ---------------- Trades overlay ---------------- #

    for trade in st.session_state.trades:
        if trade["ticker"] == ticker:
            color = "green" if trade["type"] == "BUY" else "red"

            fig.add_trace(go.Scatter(
                x=[trade["date"]],
                y=[trade["price"]],
                mode="markers",
                marker=dict(size=12, color=color),
                name=trade["type"]
            ))

            ymin = min(ymin, trade["price"])
            ymax = max(ymax, trade["price"])

    # ---------------- Forecast ---------------- #

    if show_forecast:
        f_dates, f_prices = linear_forecast(df)

        if f_dates is not None:
            fig.add_trace(go.Scatter(
                x=f_dates,
                y=f_prices,
                name="Forecast",
                line=dict(dash="dot", width=2)
            ))

            ymin = min(ymin, float(np.min(f_prices)))
            ymax = max(ymax, float(np.max(f_prices)))

    # padding so signals never touch edges
    padding = (ymax - ymin) * 0.05
    ymin -= padding
    ymax += padding

    fig.update_layout(
        height=550,
        template="plotly_dark",
        title=ticker,
        yaxis=dict(range=[ymin, ymax]),
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)

    if show_rsi:
        st.subheader("RSI")
        st.line_chart(df["RSI"])

    if show_macd:
        st.subheader("MACD")
        st.line_chart(df[["MACD", "Signal"]])

# ---------------- SIDEBAR ---------------- #

st.sidebar.title("📊 Trading Controls")

ticker = st.sidebar.text_input("Ticker", "AAPL")

period = st.sidebar.selectbox(
    "Time Range",
    ["1mo", "3mo", "6mo", "1y", "3y", "5y"]
)

chart_type = st.sidebar.radio("Chart Type", ["Line", "Candles"])

show_rsi = st.sidebar.checkbox("Show RSI", True)
show_macd = st.sidebar.checkbox("Show MACD", True)
show_forecast = st.sidebar.checkbox("Show Forecast", True)

qty = st.sidebar.number_input("Shares", min_value=1, value=1)

# ---------------- MAIN ---------------- #

df = load_data(ticker, period)

if df.empty:
    st.warning("No data found")
    st.stop()

df = add_indicators(df)

current_price = float(df["Close"].iloc[-1])

st.metric("Current Price", f"${current_price:,.2f}")

# ---------------- TRADING ---------------- #

col1, col2 = st.sidebar.columns(2)

if col1.button("📈 BUY"):
    cost = current_price * qty
    if st.session_state.cash >= cost:
        st.session_state.cash -= cost
        st.session_state.trades.append({
            "ticker": ticker,
            "type": "BUY",
            "price": current_price,
            "qty": qty,
            "date": datetime.now()
        })

if col2.button("📉 SELL"):
    st.session_state.cash += current_price * qty
    st.session_state.trades.append({
        "ticker": ticker,
        "type": "SELL",
        "price": current_price,
        "qty": qty,
        "date": datetime.now()
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

for trade in st.session_state.trades:
    if trade["ticker"] == ticker:
        if trade["type"] == "BUY":
            profit += (current_price - trade["price"]) * trade["qty"]
        else:
            profit += (trade["price"] - current_price) * trade["qty"]

st.metric("Cash Balance", f"${st.session_state.cash:,.2f}")
st.metric("Profit / Loss", f"${profit:,.2f}")

if st.session_state.trades:
    st.dataframe(pd.DataFrame(st.session_state.trades))
