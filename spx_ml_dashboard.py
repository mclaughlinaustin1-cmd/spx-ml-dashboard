import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import plotly.graph_objects as go

st.set_page_config("Stock AI Predictor", layout="wide")

@st.cache_data(ttl=3600)
def fetch_data(ticker, days):
    end = datetime.today()
    start = end - timedelta(days=days)
    return yf.download(ticker, start=start, end=end)

def add_features(df, price):
    df["MA20"] = df[price].rolling(20).mean()
    df["MA50"] = df[price].rolling(50).mean()
    df["Returns"] = df[price].pct_change()
    df["Ordinal"] = df.index.map(datetime.toordinal)
    df = df.dropna()
    return df

def train_models(X, y):
    lr = LinearRegression().fit(X, y)
    rf = RandomForestRegressor(n_estimators=200).fit(X, y)
    return lr, rf

def plot_chart(df, price, preds):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[price], name="Price"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], name="MA50"))

    for label, val, date in preds:
        fig.add_trace(go.Scatter(
            x=[date], y=[val], mode="markers+text",
            name=label, text=[f"${val:.2f}"]
        ))
    st.plotly_chart(fig, use_container_width=True)

# ---------- UI ----------

st.title("📊 AI Stock Predictor")

with st.sidebar:
    ticker = st.text_input("Stock Ticker", "AAPL")
    days = st.selectbox("History Range", [180, 365, 730, 1825])
    future_date = st.date_input("Predict Date", datetime.today()+timedelta(days=7))
    run = st.button("Run AI Forecast")

if run:
    data = fetch_data(ticker, days)

    price = "Close"
    data = add_features(data, price)

    X = data[["Ordinal", "MA20", "MA50", "Returns"]]
    y = data[price]

    split = int(len(X)*0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    lr, rf = train_models(X_train, y_train)

    future_features = X.iloc[-1].copy()
    future_features["Ordinal"] = future_date.toordinal()
    future_features = np.array(future_features).reshape(1, -1)

    pred_lr = lr.predict(future_features)[0]
    pred_rf = rf.predict(future_features)[0]

    st.metric("Linear Forecast", f"${pred_lr:.2f}")
    st.metric("Random Forest Forecast", f"${pred_rf:.2f}")

    st.write("Model Accuracy")
    st.write("LR:", r2_score(y_test, lr.predict(X_test)))
    st.write("RF:", r2_score(y_test, rf.predict(X_test)))

    plot_chart(data, price, [
        ("LR Prediction", pred_lr, future_date),
        ("RF Prediction", pred_rf, future_date)
    ])
