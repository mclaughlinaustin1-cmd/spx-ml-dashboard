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
    st.session_state.cash = 10000

# ---------------- DATA ---------------- #

@st.cache_data
def load_data(ticker, period):
    df = yf.download(ticker, period=period, interval="1d")
    df.dropna(inplace=True)
    return df

# ---------------- INDICATORS ---------------- #

def indicators(df):
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - 100/(1+rs)

    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9).mean()

    return df

# ---------------- FORECAST ---------------- #

def forecast(df, days):
    X = np.arange(len(df)).reshape(-1,1)
    y = df["Close"].values
    model = LinearRegression().fit(X,y)
    future = np.arange(len(df), len(df)+days).reshape(-1,1)
    preds = model.predict(future)
    future_dates = pd.date_range(df.index[-1], periods=days+1)[1:]
    return future_dates, preds

# ---------------- PLOT ---------------- #

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
        fig.add_trace(go.Scatter(x=df.index,y=df["Close"],name="Price"))

    ymin = df["Low"].min() - 1
    ymax = df["High"].max() + 1

    # Trades overlay
    for t in st.session_state.trades:
        if t["ticker"] == ticker:
            color = "green" if t["type"]=="BUY" else "red"
            fig.add_trace(go.Scatter(
                x=[t["date"]],
                y=[t["price"]],
                mode="markers",
                marker=dict(size=12,color=color),
                name=t["type"]
            ))
            ymin = min(ymin, t["price"]-1)
            ymax = max(ymax, t["price"]+1)

    if show_forecast:
        fd, fp = forecast(df, 30)
        fig.add_trace(go.Scatter(x=fd,y=fp,name="Forecast",line=dict(dash="dot")))
        ymin = min(ymin, min(fp)-1)
        ymax = max(ymax, max(fp)+1)

    fig.update_layout(
        height=550,
        xaxis_rangeslider_visible=False,
        yaxis=dict(range=[ymin,ymax]),
        template="plotly_dark",
        title=ticker
    )

    st.plotly_chart(fig, use_container_width=True)

    if show_rsi:
        st.line_chart(df["RSI"])

    if show_macd:
        st.line_chart(df[["MACD","Signal"]])

# ---------------- SIDEBAR ---------------- #

st.sidebar.title("Trading Panel")

ticker = st.sidebar.text_input("Ticker", "AAPL")

period = st.sidebar.selectbox(
    "Time Range",
    ["1mo","3mo","6mo","1y","3y","5y"]
)

chart_type = st.sidebar.radio("Chart",["Line","Candles"])

show_rsi = st.sidebar.checkbox("RSI",True)
show_macd = st.sidebar.checkbox("MACD",True)
show_forecast = st.sidebar.checkbox("Forecast",True)

qty = st.sidebar.number_input("Shares",1,1000,1)

# ---------------- TRADE EXECUTION ---------------- #

if st.sidebar.button("📈 BUY"):
    price = float(st.session_state.last_price)
    cost = price * qty
    if st.session_state.cash >= cost:
        st.session_state.cash -= cost
        st.session_state.trades.append({
            "ticker":ticker,
            "type":"BUY",
            "price":price,
            "qty":qty,
            "date":datetime.now()
        })

if st.sidebar.button("📉 SELL"):
    price = float(st.session_state.last_price)
    st.session_state.cash += price * qty
    st.session_state.trades.append({
        "ticker":ticker,
        "type":"SELL",
        "price":price,
        "qty":qty,
        "date":datetime.now()
    })

# ---------------- MAIN ---------------- #

df = load_data(ticker, period)
df = indicators(df)

st.session_state.last_price = df["Close"].iloc[-1]

plot_chart(
    df,
    ticker,
    chart_type,
    show_rsi,
    show_macd,
    show_forecast
)

# ---------------- PORTFOLIO ---------------- #

st.subheader("📊 Portfolio")

pl = 0
for t in st.session_state.trades:
    if t["ticker"] == ticker:
        direction = 1 if t["type"]=="BUY" else -1
        pl += direction * (df["Close"].iloc[-1]-t["price"]) * t["qty"]

st.metric("Cash", f"${st.session_state.cash:,.2f}")
st.metric("Profit / Loss", f"${pl:,.2f}")

if st.session_state.trades:
    st.dataframe(pd.DataFrame(st.session_state.trades))


