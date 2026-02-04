import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("📈 Stock Price Predictor (5-Year Historical Data)")

try:
    # --- USER INPUT ---
    ticker = st.text_input("Enter a stock ticker (e.g., AAPL, MSFT)", value="AAPL").upper()
    predict_date_str = st.text_input(
        "Enter a date to predict (YYYY-MM-DD)",
        value=(datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    )

    # --- RUN PREDICTION BUTTON ---
    if st.button("Run Prediction"):
        with st.spinner(f"Downloading data for {ticker}..."):
            end_date = datetime.today()
            start_date = end_date - timedelta(days=5*365)
            data = yf.download(ticker, start=start_date, end=end_date)

        if data.empty:
            st.error(f"No historical data found for ticker '{ticker}'. Try another ticker.")
        else:
            # --- CLEAN DATA ---
            data = data.dropna()
            data['Date'] = data.index
            data['Date_ordinal'] = pd.to_datetime(data['Date']).map(datetime.toordinal)
            
            X = data[['Date_ordinal']]
            y = data['Close']

            # --- SPLIT DATA ---
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            # --- TRAIN MODELS ---
            lr_model = LinearRegression()
            lr_model.fit(X_train, y_train)

            dt_model = DecisionTreeRegressor(max_depth=5, random_state=42)
            dt_model.fit(X_train, y_train)

            rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
            rf_model.fit(X_train, y_train)

            # --- TEST MODELS ---
            y_pred_lr = lr_model.predict(X_test)
            y_pred_dt = dt_model.predict(X_test)
            y_pred_rf = rf_model.predict(X_test)

            st.subheader("Model Accuracy (R² score)")
            st.write(f"Linear Regression:  {r2_score(y_test, y_pred_lr):.4f}")
            st.write(f"Decision Tree:      {r2_score(y_test, y_pred_dt):.4f}")
            st.write(f"Random Forest:      {r2_score(y_test, y_pred_rf):.4f}")

            # --- PREDICT USER DATE ---
            try:
                future_date = datetime.strptime(predict_date_str, "%Y-%m-%d")
                future_ordinal = np.array([[future_date.toordinal()]])
                
                pred_lr = float(lr_model.predict(future_ordinal)[0])
                pred_dt = float(dt_model.predict(future_ordinal)[0])
                pred_rf = float(rf_model.predict(future_ordinal)[0])

                st.subheader(f"Predicted {ticker} Close Price on {future_date.date()}")
                st.write(f"Linear Regression:  ${pred_lr:.2f}")
                st.write(f"Decision Tree:      ${pred_dt:.2f}")
                st.write(f"Random Forest:      ${pred_rf:.2f}")
            except Exception as e:
                st.error(f"Error predicting price: {e}")

except Exception as e:
    st.error(f"App encountered an error: {e}")
