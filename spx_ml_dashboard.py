import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")

# ================= STATE ================= #

if "trades" not in st.session_state:
    st.session_state.trades = []

if "cash" not in st.session_state:
    st.session_state.cash = 10000.0

# ================= DATA ================= #

@st.cache_data
def load_data(ticker, period):
    df = yf.download(ticker, period=period, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    return df.dropna()

# ================= INDICATORS ================= #

def indicators(df):
    close = df["Close"]

    df["MA20"] = close.rolling(20).mean()
    df["MA50"] = close.rolling(50).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()

    rs = gain / loss
    df["RSI"] = 100 - 100/(1+rs)

    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()

    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9).mean()

    return df

# ================= FORECAST ================= #

def forecast(df, days=30):
    if len(df) < 30:
        return None, None

    y = df["Close"].values
    X = np.arange(len(y)).reshape(-1,1)

    model = LinearRegression().fit(X, y)

    future_X = np.arange(len(y), len(y)+days).reshape(-1,1)
    preds = model.predict(future_X)

    dates = pd.date_range(df.index[-1], periods=days+1)[1:]
    return dates, preds

# ================= BACKTEST ================= #

def backtest(df):
    df = df.copy()
    df["Position"] = np.where(df["MA20"] > df["MA50"], 1, 0)
    df["Return"] = df["Close"].pct_change()
    df["Strategy"] = df["Return"] * df["Position"].shift()

    df["Equity"] = (1 + df["Strategy"]).cumprod()

    trades = []
    entry = None

    for i in range(1, len(df)):
        if df["Position"].iloc[i] == 1 and df["Position"].iloc[i-1] == 0:
            entry = df["Close"].iloc[i]
        if df["Position"].iloc[i] == 0 and df["Position"].iloc[i-1] == 1 and entry:
            exit_price = df["Close"].iloc[i]
            trades.append(exit_price - entry)
            entry = None

    wins = sum(1 for t in trades if t > 0)
    losses = len(trades) - wins

    win_rate = wins / len(trades) * 100 if trades else 0
    total_return = (df["Equity"].iloc[-1]-1)*100

    peak = df["Equity"].cummax()
    drawdown = ((df["Equity"] - peak) / peak).min() * 100

    return df, win_rate, total_return, drawdown, trades

# ================= CHART ================= #

def draw_chart(df, ticker, candles, show_forecast):

    fig = go.Figure()

    if candles:
        fig.add_candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"]
        )
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price"))

    fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], name="MA50"))

    if show_forecast:
        fd, fp = forecast(df)
        if fd is not None:
            fig.add_trace(go.Scatter(
                x=fd,
                y=fp,
                line=dict(dash="dot"),
                name="Forecast"
            ))

    fig.update_layout(
        template="plotly_dark",
        height=600,
        xaxis_rangeslider_visible=False,
        title=ticker
    )

    st.plotly_chart(fig, use_container_width=True)

# ================= SIDEBAR ================= #

st.sidebar.title("⚙ Controls")

ticker = st.sidebar.text_input("Ticker", "AAPL").upper()
period = st.sidebar.selectbox("Range", ["3mo","6mo","1y","3y","5y"])
candles = st.sidebar.checkbox("Candlesticks", True)
show_forecast = st.sidebar.checkbox("AI Forecast", True)

df = load_data(ticker, period)

if df.empty:
    st.stop()

df = indicators(df)

price = float(df["Close"].iloc[-1])

# ================= TABS ================= #

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Chart & Indicators",
    "📈 Paper Trading",
    "💼 Portfolio",
    "🧪 Backtesting & Analytics"
])

# ---------- CHART ---------- #

with tab1:
    draw_chart(df, ticker, candles, show_forecast)

    st.subheader("RSI")
    st.line_chart(df["RSI"])

    st.subheader("MACD")
    st.line_chart(df[["MACD","Signal"]])

# ---------- SIM ---------- #

with tab2:
    qty = st.number_input("Shares", 1, 1000, 1)

    c1, c2 = st.columns(2)

    if c1.button("BUY"):
        cost = price * qty
        if st.session_state.cash >= cost:
            st.session_state.cash -= cost
            st.session_state.trades.append(("BUY", price, qty))

    if c2.button("SELL"):
        st.session_state.cash += price * qty
        st.session_state.trades.append(("SELL", price, qty))

    if st.session_state.trades:
        st.dataframe(pd.DataFrame(st.session_state.trades, columns=["Type","Price","Qty"]))

# ---------- PORTFOLIO ---------- #

with tab3:
    profit = 0
    for t in st.session_state.trades:
        if t[0] == "BUY":
            profit += (price - t[1]) * t[2]
        else:
            profit += (t[1] - price) * t[2]

    st.metric("Cash", f"${st.session_state.cash:,.2f}")
    st.metric("Profit/Loss", f"${profit:,.2f}")

# ---------- BACKTEST ---------- #

with tab4:
    bt, win_rate, total_return, drawdown, trades = backtest(df)

    col1, col2, col3 = st.columns(3)

    col1.metric("Win Rate", f"{win_rate:.1f}%")
    col2.metric("Total Return", f"{total_return:.1f}%")
    col3.metric("Max Drawdown", f"{drawdown:.1f}%")

    st.subheader("Equity Curve")
    st.line_chart(bt["Equity"])

    st.subheader("Trade Results")
    if trades:
        st.bar_chart(pd.Series(trades))
    else:
        st.info("Not enough signals yet")
