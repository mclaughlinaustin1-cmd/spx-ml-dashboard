import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from datetime import datetime, timedelta
import plotly.graph_objects as go

# --- Page Config ---
st.set_page_config(layout="wide", page_title="ML Price Predictor", page_icon="🤖")

st.title("🤖 ML Institutional Terminal: Triple-Model Analysis")
st.markdown("""
This terminal trains three distinct models—**Linear**, **Non-Linear (Polynomial)**, and **Decision Trees**—using 5 years of historical data to project prices for a user-defined date.
""")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("📋 Analysis Settings")
    ticker = st.text_input("Ticker Symbol", "AAPL").upper()
    
    # User prompt for specific prediction date
    st.subheader("🔮 Predict Future Date")
    target_date = st.date_input("Select Date for Prediction", datetime.now() + timedelta(days=30))
    
    st.divider()
    st.info("The model uses the last 5 years of data for training.")

# --- Data Engine ---
@st.cache_data(ttl=3600)
def get_ml_data(symbol):
    # Fetch 5 years of data as requested
    df = yf.download(symbol, period="5y", interval="1d")
    if df.empty: return None
    
    # Data Cleaning
    df = df[['Close']].copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    
    # Create time-step features for ML
    df['Day_Index'] = np.arange(len(df))
    return df

df = get_ml_data(ticker)

if df is not None:
    # --- Training & Accuracy Testing ---
    X = df[['Day_Index']].values
    y = df['Close'].values
    
    # 1. Linear Regression
    lin_model = LinearRegression()
    lin_model.fit(X, y)
    lin_acc = r2_score(y, lin_model.predict(X))

    # 2. Non-Linear Regression (Polynomial Degree 2)
    poly_features = PolynomialFeatures(degree=2)
    X_poly = poly_features.fit_transform(X)
    poly_model = LinearRegression()
    poly_model.fit(X_poly, y)
    poly_acc = r2_score(y, poly_model.predict(X_poly))

    # 3. Decision Tree Regressor
    tree_model = DecisionTreeRegressor(max_depth=5)
    tree_model.fit(X, y)
    tree_acc = r2_score(y, tree_model.predict(X))

    # --- Accuracy Metrics Display ---
    st.subheader("🎯 Model Accuracy (R² Score)")
    a1, a2, a3 = st.columns(3)
    a1.metric("Linear Regression", f"{lin_acc:.4f}")
    a2.metric("Non-Linear (Poly)", f"{poly_acc:.4f}")
    a3.metric("Decision Tree", f"{tree_acc:.4f}")

    # --- Prediction Logic for Target Date ---
    # Calculate days between last data point and target date
    days_delta = (pd.Timestamp(target_date) - df.index[-1]).days
    target_index = len(df) + days_delta
    X_target = np.array([[target_index]])

    # Generate Predictions
    pred_lin = lin_model.predict(X_target)[0]
    pred_poly = poly_model.predict(poly_features.transform(X_target))[0]
    pred_tree = tree_model.predict(X_target)[0]

    st.divider()
    st.subheader(f"💰 Price Predictions for {target_date}")
    p1, p2, p3 = st.columns(3)
    p1.subheader(f"Linear: **${pred_lin:,.2f}**")
    p2.subheader(f"Non-Linear: **${pred_poly:,.2f}**")
    p3.subheader(f"Decision Tree: **${pred_tree:,.2f}**")

    # --- Visualization ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Actual Price", line=dict(color="white", width=1)))
    
    # Forecast Lines
    future_range = pd.date_range(start=df.index[-1], end=target_date)
    future_indices = np.arange(len(df), len(df) + len(future_range)).reshape(-1, 1)
    
    fig.add_trace(go.Scatter(x=future_range, y=poly_model.predict(poly_features.transform(future_indices)), 
                             name="Non-Linear Trend", line=dict(color="#00ffcc", dash="dash")))
    
    fig.update_layout(template="plotly_dark", height=500, title=f"{ticker} Historical & Projected Path")
    st.plotly_chart(fig, use_container_width=True)

    

else:
    st.error("Invalid Ticker or No Data Found.")

st.markdown("""
---
**Technical Note:** * **Linear Regression** assumes a constant rate of growth.
* **Non-Linear (Polynomial)** captures the curvature of market momentum.
* **Decision Trees** split data into segments to find patterns but can be prone to "flat" forecasts in the future.
""")


