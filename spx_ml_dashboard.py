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
import plotly.graph_objects as go

st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("📈 Stock Price Predictor & Interactive Chart")

try:
    # --- USER INPUT ---
    ticker = st.text_input("Enter a stock ticker (e.g., AAPL, MSFT)", value="AAPL").upper()

    range_options = {
        "24 hours": 1,
        "6 months": 182,
        "1 year": 365,
        "2 years": 730,
        "5 years": 1825
    }
    selected_range = st.selectbox("Select historical data range:", options=list(range_options.keys()))
    predict_date_str = st.text_input(
        "Enter a date to predict (YYYY-MM-DD)",
        value=(datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    )

    if st.button("Run Prediction"):
        # --- DOWNLOAD DATA ---
        end_date = datetime.today()
        start_date = end_date - timedelta(days=range_options[selected_range])
        data = yf.download(ticker, start=start_date, end=end_date, interval="1d")

        if data.empty:
            st.error(f"No historical data found for ticker '{ticker}'.")
        else:
            # --- FLATTEN MULTIINDEX COLUMNS ---
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [' '.join(col).strip() for col in data.columns.values]

            # --- SELECT PRICE COLUMN ---
            if 'Close' in data.columns:
                price_col = 'Close'
            elif 'Adj Close' in data.columns:
                price_col = 'Adj Close'
            else:
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    st.error("No numeric price column found.")
                    st.stop()
                price_col = numeric_cols[0]

            data = data.dropna(subset=[price_col])
            data['Date'] = data.index
            data['Date_ordinal'] = pd.to_datetime(data['Date']).map(datetime.toordinal)

            X = data[['Date_ordinal']]
            y = data[price_col]

            # --- SPLIT AND TRAIN MODELS ---
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            lr_model = LinearRegression().fit(X_train, y_train)
            dt_model = DecisionTreeRegressor(max_depth=5, random_state=42).fit(X_train, y_train)
            rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42).fit(X_train, y_train)

            # --- TEST ACCURACY ---
            st.subheader("Model Accuracy (R² score)")
            st.write(f"Linear Regression:  {r2_score(y_test, lr_model.predict(X_test)):.4f}")
            st.write(f"Decision Tree:      {r2_score(y_test, dt_model.predict(X_test)):.4f}")
            st.write(f"Random Forest:      {r2_score(y_test, rf_model.predict(X_test)):.4f}")

            # --- PREDICT FUTURE DATE ---
            try:
                future_date = datetime.strptime(predict_date_str, "%Y-%m-%d")
                future_ordinal = np.array([[future_date.toordinal()]])

                pred_lr = lr_model.predict(future_ordinal).item()
                pred_dt = dt_model.predict(future_ordinal).item()
                pred_rf = rf_model.predict(future_ordinal).item()

                st.subheader(f"Predicted {ticker} Close Price on {future_date.date()}")
                st.write(f"Linear Regression:  ${pred_lr:.2f}")
                st.write(f"Decision Tree:      ${pred_dt:.2f}")
                st.write(f"Random Forest:      ${pred_rf:.2f}")

            except Exception as e:
                st.error(f"Error predicting price: {e}")

            # --- INTERACTIVE PLOTLY CHART ---
            fig = go.Figure()
            # Historical prices
            fig.add_trace(go.Scatter(
                x=data['Date'], y=data[price_col],
                mode='lines+markers', name='Historical Price'
            ))
            # Predicted price
            if 'future_date' in locals():
                fig.add_trace(go.Scatter(
                    x=[future_date], y=[pred_lr],
                    mode='markers+text', name='Predicted (LR)',
                    marker=dict(color='green', size=12),
                    text=[f"${pred_lr:.2f}"]
                ))
                fig.add_trace(go.Scatter(
                    x=[future_date], y=[pred_dt],
                    mode='markers+text', name='Predicted (DT)',
                    marker=dict(color='orange', size=12),
                    text=[f"${pred_dt:.2f}"]
                ))
                fig.add_trace(go.Scatter(
                    x=[future_date], y=[pred_rf],
                    mode='markers+text', name='Predicted (RF)',
                    marker=dict(color='red', size=12),
                    text=[f"${pred_rf:.2f}"]
                ))

            fig.update_layout(
                title=f"{ticker} Historical & Predicted Price",
                xaxis_title="Date", yaxis_title="Price",
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"App encountered an error: {e}")
