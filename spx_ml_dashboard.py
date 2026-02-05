import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objects as go

# ----------------- Page Setup -----------------
st.set_page_config(page_title="Ultimate AI Trading Dashboard", layout="wide")
st.title("🚀 Ultimate AI Trading + Candlestick + Paper Trading Simulator")

# ----------------- Data Loader -----------------
@st.cache_data(ttl=1800)
def load_data(ticker, days):
    end = datetime.today()
    start = end - timedelta(days=days)
    interval = "1h" if days <= 30 else "1d"
    df = yf.download(ticker, start=start, end=end, interval=interval)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

# ----------------- Indicators -----------------
def add_indicators(df):
    df = df.copy()
    close = df["Close"]
    if len(close) < 14:
        df["RSI"] = np.nan
        df["MACD"] = np.nan
        df["Signal"] = np.nan
        df["Volatility"] = np.nan
        df["Unusual"] = False
        return df
    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    # Volatility
    df["Volatility"] = close.pct_change().rolling(20).std()
    # Whale proxy
    df["Unusual"] = df["Volume"] > df["Volume"].rolling(20).mean()*2
    return df.dropna()

# ----------------- Trend Probability -----------------
def trend_probability(df):
    if len(df) < 10: return 0.5
    df["Future"] = df["Close"].shift(-3)
    df["Target"] = (df["Future"] > df["Close"]).astype(int)
    df = df.dropna()
    features = ["RSI","MACD","Signal","Volatility"]
    X = df[features].dropna()
    y = df.loc[X.index,"Target"]
    if len(X) < 1: return 0.5
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_scaled, y)
    latest = df[features].iloc[-1:].fillna(method="bfill")
    latest_scaled = scaler.transform(latest)
    return model.predict_proba(latest_scaled)[0][1]

# ----------------- Signals -----------------
def signal(rsi, macd, sig):
    if pd.isna(rsi) or pd.isna(macd) or pd.isna(sig): return "N/A"
    if rsi < 30 and macd > sig: return "BUY"
    if rsi > 70 and macd < sig: return "SELL"
    return "HOLD"

def risk_level(vol):
    if pd.isna(vol): return "N/A"
    if vol < 0.01: return "LOW"
    if vol < 0.025: return "MEDIUM"
    return "HIGH"

# ----------------- LSTM Forecast -----------------
@st.cache_data(ttl=3600)
def lstm_forecast(prices, steps=7):
    if len(prices) < 50: return []
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices.reshape(-1,1))
    X, y = [], []
    window = 30
    for i in range(window, len(scaled)-steps):
        X.append(scaled[i-window:i,0])
        y.append(scaled[i:i+steps,0])
    if len(X)==0: return []
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0],X.shape[1],1))
    model = Sequential()
    model.add(LSTM(30, activation='relu', input_shape=(X.shape[1],1)))
    model.add(Dense(steps))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X,y,epochs=5,batch_size=8,verbose=0)
    last_window = scaled[-window:].reshape((1,window,1))
    pred_scaled = model.predict(last_window, verbose=0)[0]
    return scaler.inverse_transform(pred_scaled.reshape(-1,1)).flatten()

# ----------------- Paper Trading -----------------
if "sim_cash" not in st.session_state:
    st.session_state.sim_cash = 10000
    st.session_state.sim_holdings = {}

def paper_trade(ticker, action, amount):
    df = load_data(ticker, 30)
    if df.empty: return
    price = df["Close"].iloc[-1]
    if action=="BUY":
        qty = amount / price
        if st.session_state.sim_cash >= amount:
            st.session_state.sim_cash -= amount
            if ticker not in st.session_state.sim_holdings:
                st.session_state.sim_holdings[ticker] = {'qty':0,'avg_price':0}
            current = st.session_state.sim_holdings[ticker]
            total_cost = current['qty']*current['avg_price'] + amount
            total_qty = current['qty'] + qty
            current['qty'] = total_qty
            current['avg_price'] = total_cost / total_qty
    elif action=="SELL":
        if ticker in st.session_state.sim_holdings:
            current = st.session_state.sim_holdings[ticker]
            if current['qty']>0:
                sell_qty = min(amount/price, current['qty'])
                current['qty'] -= sell_qty
                st.session_state.sim_cash += sell_qty*price

# ----------------- Plot Chart -----------------
def plot_chart(df, ticker, lstm_pred=None, chart_type="line",
               show_rsi=True, show_macd=True, show_signals=True,
               show_forecast=True, key=None, zoom=False):

    df_plot = df.iloc[-50:] if zoom else df
    fig = go.Figure()

    # Price
    if chart_type=="candlestick":
        fig.add_trace(go.Candlestick(x=df_plot.index,
                                     open=df_plot["Open"],
                                     high=df_plot["High"],
                                     low=df_plot["Low"],
                                     close=df_plot["Close"],
                                     name="Price"))
    else:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["Close"], mode="lines", name="Price"))

    # Indicators
    if show_macd and "MACD" in df_plot:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["MACD"], name="MACD"))
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["Signal"], name="Signal"))
    if show_rsi and "RSI" in df_plot:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["RSI"], name="RSI"))

    # Signals
    buy_idx = df_plot[(df_plot["RSI"]<30) & (df_plot["MACD"]>df_plot["Signal"])].index
    sell_idx = df_plot[(df_plot["RSI"]>70) & (df_plot["MACD"]<df_plot["Signal"])].index
    fig.add_trace(go.Scatter(x=buy_idx, y=df_plot.loc[buy_idx,"Close"], mode="markers",
                             marker=dict(size=12,color="green"), name="BUY"))
    fig.add_trace(go.Scatter(x=sell_idx, y=df_plot.loc[sell_idx,"Close"], mode="markers",
                             marker=dict(size=12,color="red"), name="SELL"))

    # Whale Proxy
    unusual_idx = df_plot[df_plot["Unusual"]].index
    fig.add_trace(go.Scatter(x=unusual_idx, y=df_plot.loc[unusual_idx,"Close"], mode="markers",
                             marker=dict(size=14,color="purple",symbol="diamond"), name="Whale Proxy"))

    # LSTM Forecast
    if show_forecast and lstm_pred is not None:
        future_dates = [df_plot.index[-1] + pd.Timedelta(days=i+1) for i in range(len(lstm_pred))]
        fig.add_trace(go.Scatter(x=future_dates, y=lstm_pred, mode="lines+markers", name="LSTM Forecast"))

    # Autoscale Y-axis: $1 buffer above/below all points
    all_y = df_plot["Close"].tolist()
    if lstm_pred is not None: all_y += lstm_pred.tolist()
    if len(buy_idx)>0: all_y += df_plot.loc[buy_idx,"Close"].tolist()
    if len(sell_idx)>0: all_y += df_plot.loc[sell_idx,"Close"].tolist()
    if len(unusual_idx)>0: all_y += df_plot.loc[unusual_idx,"Close"].tolist()
    y_min, y_max = min(all_y)-1, max(all_y)+1

    # X-axis autoscale (zoomed or full)
    x_min = df_plot.index.min()
    x_max = df_plot.index.max()

    fig.update_layout(
        title=f"{ticker} Price & Signals",
        hovermode="x unified",
        xaxis=dict(range=[x_min, x_max], rangeslider=dict(visible=True)),
        yaxis=dict(range=[y_min, y_max], fixedrange=False)
    )
    st.plotly_chart(fig, use_container_width=True, key=key)

# ----------------- Sidebar -----------------
with st.sidebar:
    tickers_input = st.text_input("Tickers (comma)", "AAPL,MSFT,GOOG")
    range_options = {"24 hours":1,"1 week":7,"1 month":30,"6 months":182,"1 year":365,"3 years":1095,"5 years":1825}
    selected_range = st.selectbox("Historical range:", options=list(range_options.keys()))
    days = range_options[selected_range]
    show_rsi = st.checkbox("Show RSI", value=True)
    show_macd = st.checkbox("Show MACD", value=True)
    show_signals = st.checkbox("Show Signals", value=True)
    show_forecast = st.checkbox("Show Forecast", value=True)
    chart_type = st.radio("Chart Type", ["line","candlestick"])
    run = st.button("Run Platform")

    st.subheader("ℹ️ Indicator Guide")
    st.markdown("""
    - **Price:** Close / candlestick  
    - **RSI:** 0-100, <30 BUY, >70 SELL  
    - **MACD:** Trend direction  
    - **Signal:** EMA9 of MACD  
    - **Trend Probability:** AI chance to rise  
    - **Volatility:** Risk  
    - **Buy/Sell Signals:** Generated by RSI+MACD  
    - **Whale Proxy:** Large volume spike  
    - **LSTM Forecast:** AI future price
    """)

# ----------------- Main -----------------
st.subheader("📊 Market Overview")
if run:
    tickers = [t.strip().upper() for t in tickers_input.split(",")]
    for t in tickers:
        df = load_data(t, days)
        if df.empty:
            st.warning(f"{t}: No data available")
            continue
        df = add_indicators(df)
        prob_up = trend_probability(df) if not df.empty else 0.5
        sig = signal(df["RSI"].iloc[-1], df["MACD"].iloc[-1], df["Signal"].iloc[-1]) if not df.empty else "N/A"
        risk = risk_level(df["Volatility"].iloc[-1]) if not df.empty else "N/A"
        lstm_pred = lstm_forecast(df["Close"].values) if len(df["Close"])>=50 else []

        tab1, tab2, tab3 = st.tabs([f"{t} Full Chart", f"{t} Zoom", f"{t} Paper Trading"])
        with tab1:
            plot_chart(df, t, lstm_pred, chart_type, zoom=False,
                       show_rsi=show_rsi, show_macd=show_macd,
                       show_signals=show_signals, show_forecast=show_forecast,
                       key=f"{t}_full")
        with tab2:
            plot_chart(df, t, lstm_pred, chart_type, zoom=True,
                       show_rsi=show_rsi, show_macd=show_macd,
                       show_signals=show_signals, show_forecast=show_forecast,
                       key=f"{t}_zoom")
        with tab3:
            st.header(f"🧪 Paper Trading Simulator for {t}")
            sim_action = st.radio("Action", ["BUY","SELL"], key=f"{t}_action")
            sim_amount = st.number_input("Amount to trade ($)", 100, step=100, key=f"{t}_amt")
            if st.button("Execute Trade", key=f"{t}_btn"):
                paper_trade(t, sim_action, sim_amount)
            st.subheader("💼 Portfolio Overview")
            st.write("Cash:", round(st.session_state.sim_cash,2))
            holdings_list = []
            for h_t,h in st.session_state.sim_holdings.items():
                price = load_data(h_t, 30)["Close"].iloc[-1] if not load_data(h_t,30).empty else h['avg_price']
                pl_percent = (price - h['avg_price'])/h['avg_price']*100 if h['avg_price']>0 else 0
                holdings_list.append([h_t,h['qty'],h['avg_price'],price,round(pl_percent,2)])
            st.dataframe(pd.DataFrame(holdings_list, columns=["Ticker","Qty","Avg Price","Current Price","P/L %"]))

        if sig=="BUY" and prob_up>0.6: st.info(f"🚨 {t} strong BUY ({round(prob_up*100,1)}%)")
        if sig=="SELL" and prob_up<0.4: st.warning(f"⚠ {t} strong SELL ({round(prob_up*100,1)}%)")
