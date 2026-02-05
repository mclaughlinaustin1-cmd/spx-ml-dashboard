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
    df = yf.download(ticker, start=start, end=end)

    # Flatten multi-index columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df[["Close"]].dropna()

# ---------------- FEATURES ----------------

def make_features(df):
    price = df["Close"].values.ravel()  # force 1D

    ma20 = pd.Series(price).rolling(20).mean().values
    ma50 = pd.Series(price).rolling(50).mean().values
    returns = pd.Series(price).pct_change().values

    X = np.column_stack([
        np.arange(len(price)),
        ma20,
        ma50,
        returns
    ])

    mask = ~np.isnan(X).any(axis=1)

    X = X[mask]
    y = price[mask]
    dates = df.index[mask]

    return X, y, dates

# ---------------- MODEL ----------------

def train_models(X, y):
    lr = LinearRegression()
    rf = RandomForestRegressor(n_estimators=250, random_state=42)

    lr.fit(X, y)
    rf.fit(X, y)

    return lr, rf

# ---------------- PLOT ----------------

def plot_chart(dates, prices, preds, future_date):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=dates, y=prices, name="Price"))

    fig.add_trace(go.Scatter(
        x=[future_date],
        y=[preds["lr"]],
        mode="markers+text",
        name="LR Prediction",
        text=[f"${preds['lr']:.2f}"]
    ))

    fig.add_trace(go.Scatter(
        x=[future_date],
        y=[preds["rf"]],
        mode="markers+text",
        name="RF Prediction",
        text=[f"${preds['rf']:.2f}"]
    ))

    fig.update_layout(
        title="Stock Price & AI Forecast",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------- UI ----------------

with st.sidebar:
    ticker = st.text_input("Stock ticker", "AAPL").upper()
    days = st.selectbox("History range", [180, 365, 730, 1825], index=1)
    future_date = st.date_input("Prediction date", datetime.today()+timedelta(days=7))
    run = st.button("Run Forecast")

# ---------------- APP ----------------

if run:
    raw = fetch_data(ticker, days)

    if raw.empty or len(raw) < 80:
        st.error("Not enough stock data.")
        st.stop()

    X, y, dates = make_features(raw)

    split = int(len(X) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    lr, rf = train_models(X_train, y_train)

    last_prices = raw["Close"].values.ravel()

    future_features = np.array([
        len(last_prices),
        np.mean(last_prices[-20:]),
        np.mean(last_prices[-50:]),
        (last_prices[-1] - last_prices[-2]) / last_prices[-2]
    ]).reshape(1, -1)

    pred_lr = lr.predict(future_features).item()
    pred_rf = rf.predict(future_features).item()

    col1, col2, col3 = st.columns(3)
    col1.metric("Linear Regression", f"${pred_lr:.2f}")
    col2.metric("Random Forest", f"${pred_rf:.2f}")
    col3.metric("Average", f"${(pred_lr + pred_rf)/2:.2f}")

    st.subheader("Model Accuracy")
    st.write("LR R²:", round(r2_score(y_test, lr.predict(X_test)), 4))
    st.write("RF R²:", round(r2_score(y_test, rf.predict(X_test)), 4))

    plot_chart(dates, last_prices[-len(dates):], {"lr": pred_lr, "rf": pred_rf}, future_date)

    st.success("Forecast complete!")
