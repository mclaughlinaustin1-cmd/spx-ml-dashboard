import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

st.set_page_config(layout="wide")

# ================= DATA ================= #

@st.cache_data
def load_data(ticker, period):
    df = yf.download(ticker, period=period, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    return df.dropna()

# ================= FEATURES ================= #

def features(df):
    df = df.copy()

    df["Return"] = df["Close"].pct_change()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA30"] = df["Close"].rolling(30).mean()
    df["Vol"] = df["Return"].rolling(10).std()

    df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)

    return df.dropna()

# ================= AI TREND MODEL ================= #

def train_ai(df):
    X = df[["Return","MA10","MA30","Vol"]]
    y = df["Target"]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

    model = RandomForestClassifier(n_estimators=300)
    model.fit(X_train, y_train)

    prob = model.predict_proba(X)[:,1]

    df["TrendProb"] = prob
    df["Signal"] = np.where(prob > 0.55, 1, 0)

    return df

# ================= LSTM FORECAST ================= #

def lstm_forecast(df, steps=30):
    prices = df["Close"].values.reshape(-1,1)

    if len(prices) < 60:
        return None, None

    scaler = StandardScaler()
    prices_scaled = scaler.fit_transform(prices)

    X, y = [], []
    for i in range(50, len(prices_scaled)):
        X.append(prices_scaled[i-50:i])
        y.append(prices_scaled[i])

    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(50,1)),
        LSTM(32),
        Dense(1)
    ])

    model.compile(optimizer=Adam(0.01), loss="mse")
    model.fit(X, y, epochs=5, verbose=0)

    last = X[-1]
    preds = []

    for _ in range(steps):
        p = model.predict(last.reshape(1,50,1), verbose=0)[0]
        preds.append(p)
        last = np.vstack([last[1:], p])

    preds = scaler.inverse_transform(np.array(preds))

    dates = pd.date_range(df.index[-1], periods=steps+1)[1:]

    return dates, preds.flatten()

# ================= BACKTEST ================= #

def backtest(df):
    df["StrategyReturn"] = df["Return"] * df["Signal"].shift()
    df["Equity"] = (1 + df["StrategyReturn"]).cumprod()

    peak = df["Equity"].cummax()
    drawdown = ((df["Equity"] - peak) / peak).min()

    win_rate = (df["StrategyReturn"] > 0).mean() * 100
    total_return = (df["Equity"].iloc[-1] - 1) * 100

    return df, win_rate, total_return, drawdown*100

# ================= CHART ================= #

def chart(df, ticker, candles, show_lstm):

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

    buys = df[df["Signal"]==1]

    fig.add_trace(go.Scatter(
        x=buys.index,
        y=buys["Close"],
        mode="markers",
        marker=dict(color="lime", size=6),
        name="AI Buy Signal"
    ))

    if show_lstm:
        fd, fp = lstm_forecast(df)
        if fd is not None:
            fig.add_trace(go.Scatter(
                x=fd,
                y=fp,
                line=dict(dash="dot"),
                name="AI Forecast"
            ))

    fig.update_layout(
        template="plotly_dark",
        height=600,
        xaxis_rangeslider_visible=False,
        title=ticker
    )

    st.plotly_chart(fig, use_container_width=True)

# ================= UI ================= #

st.sidebar.title("🧠 AI Trading System")

ticker = st.sidebar.text_input("Ticker", "AAPL")
period = st.sidebar.selectbox("Range", ["6mo","1y","3y","5y"])
candles = st.sidebar.checkbox("Candlesticks", True)
show_lstm = st.sidebar.checkbox("Show LSTM Forecast", True)

df = load_data(ticker, period)

df = features(df)
df = train_ai(df)

tabs = st.tabs([
    "📊 AI Chart",
    "📈 Backtest Analytics",
    "🧠 Trend Probabilities"
])

# ====== CHART ====== #

with tabs[0]:
    chart(df, ticker, candles, show_lstm)

# ====== BACKTEST ====== #

with tabs[1]:
    bt, win, ret, dd = backtest(df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Win Rate", f"{win:.1f}%")
    c2.metric("Total Return", f"{ret:.1f}%")
    c3.metric("Max Drawdown", f"{dd:.1f}%")

    st.subheader("Equity Curve")
    st.line_chart(bt["Equity"])

# ====== PROB ====== #

with tabs[2]:
    st.subheader("AI Trend Probability (0–1)")

    st.line_chart(df["TrendProb"])
