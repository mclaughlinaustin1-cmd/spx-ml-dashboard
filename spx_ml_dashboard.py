import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import plotly.graph_objects as go

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Stock Predictor", layout="wide")
st.title("📈 AI Stock Price Predictor")

# ---------------- DATA ----------------

@st.cache_data(ttl=3600)
def fetch_data(ticker, days):
    end = datetime.today()
    start = end - timedelta(days=days)
    return yf.download(ticker, start=start, end=end)

# ---------------- FEATURES ----------------

def add_features(df):
    df = df.copy()
    price = "Close"

    df["MA20"] = df[price].rolling(20).mean()
    df["MA50"] = df[price].rolling(50).mean()
    df["Returns"] = df[price].pct_change()
    df["Ordinal"] = df.index.map(datetime.toordinal)

    df = df.dropna()
    return df

# ---------------- MODEL ----------------

def train_models(X, y):
    lr = LinearRegression()
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        random_state=42
    )

    lr.fit(X, y)
    rf.fit(X, y)

    return lr, rf

# ---------------- PLOT ----------------

def plot_chart(df, preds, future_date):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["Close"],
        name="Price",
        mode="lines"
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["MA20"],
        name="MA20"
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["MA50"],
        name="MA50"
    ))

    fig.add_trace(go.Scatter(
        x=[future_date],
        y=[preds["lr"]],
        mode="markers+text",
        name="Linear Prediction",
        text=[f"${preds['lr']:.2f}"],
        marker=dict(size=12)
    ))

    fig.add_trace(go.Scatter(
        x=[future_date],
        y=[preds["rf"]],
        mode="markers+text",
        name="Random Forest Prediction",
        text=[f"${preds['rf']:.2f}"],
        marker=dict(size=12)
    ))

    fig.update_layout(
        title="Historical Price & AI Predictions",
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Price"
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------- UI ----------------

with st.sidebar:
    st.header("⚙ Controls")

    ticker = st.text_input("Stock ticker", "AAPL").upper()

    days = st.selectbox(
        "Historical range",
        [180, 365, 730, 1825],
        index=1
    )

    future_date = st.date_input(
        "Prediction date",
        datetime.today() + timedelta(days=7)
    )

    run = st.button("Run AI Forecast")

# ---------------- APP ----------------

if run:
    with st.spinner("Fetching stock data..."):
        data = fetch_data(ticker, days)

    if data.empty:
        st.error("No data found for this ticker.")
        st.stop()

    if len(data) < 80:
        st.warning("Not enough historical data to train models.")
        st.stop()

    data = add_features(data)

    features = ["Ordinal", "MA20", "MA50", "Returns"]

    X = data[features]
    y = data["Close"]

    split = int(len(X) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    lr, rf = train_models(X_train, y_train)

    future_features = X.iloc[-1].copy()
    future_features["Ordinal"] = future_date.toordinal()
    future_features = future_features.values.reshape(1, -1)

    pred_lr = float(lr.predict(future_features)[0])
    pred_rf = float(rf.predict(future_features)[0])

    if np.isnan(pred_lr) or np.isnan(pred_rf):
        st.error("Prediction failed — insufficient clean data.")
        st.stop()

    col1, col2, col3 = st.columns(3)

    col1.metric("📉 Linear Regression", f"${pred_lr:.2f}")
    col2.metric("🌲 Random Forest", f"${pred_rf:.2f}")
    col3.metric("📊 Avg Forecast", f"${(pred_lr + pred_rf)/2:.2f}")

    st.subheader("Model Accuracy (R²)")

    st.write("Linear Regression:", round(r2_score(y_test, lr.predict(X_test)), 4))
    st.write("Random Forest:", round(r2_score(y_test, rf.predict(X_test)), 4))

    plot_chart(
        data,
        {"lr": pred_lr, "rf": pred_rf},
        future_date
    )

    st.success("Prediction complete!")
