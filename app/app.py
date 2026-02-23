import sys
import os

# Ensure the project root (myenv/) is on sys.path so `src` package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

from src.features import add_indicators
from src.predict import load_artifacts
from src.config import STOCKS

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Direction Predictor",
    page_icon="📈",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .big-metric { font-size: 2.4rem; font-weight: 700; }
    .green  { color: #00c853; }
    .red    { color: #ff1744; }
    div[data-testid="stMetric"] { background: #0e1117; border-radius: .6rem; padding: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")

# Load model + encoder (cached so it only runs once)
@st.cache_resource
def get_model():
    return load_artifacts()

model, encoder = get_model()

stock = st.sidebar.selectbox("Select Stock", encoder.classes_)

start = st.sidebar.date_input("Start Date", pd.to_datetime("2024-01-01"))
end = st.sidebar.date_input("End Date", pd.to_datetime("today"))

if start >= end:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

# ── Data download (cached per stock + date range) ───────────────────────────
@st.cache_data(ttl=600, show_spinner="Downloading market data …")
def fetch(ticker, s, e):
    df = yf.download(ticker, start=s, end=e, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

raw = fetch(stock, start, end)

if raw.empty:
    st.warning("No data returned for this stock / date range.")
    st.stop()

raw.reset_index(inplace=True)
raw["Stock"] = stock

# ── Feature engineering ──────────────────────────────────────────────────────
df = add_indicators(raw)

if df.empty:
    st.warning("Not enough data to compute indicators. Try a wider date range.")
    st.stop()

# ── Prediction on the latest row ────────────────────────────────────────────
latest = df.iloc[[-1]].copy()
latest["Stock"] = encoder.transform([stock])[0]

feature_cols = [
    "Open", "High", "Low", "Close", "Volume",
    "Return", "SMA_10", "SMA_20", "Volatility", "RSI", "Stock",
]
X = latest[feature_cols]

prediction = model.predict(X)[0]
prob = model.predict_proba(X)[0]
confidence = max(prob) * 100

# ── Header ───────────────────────────────────────────────────────────────────
st.title("📈 Multi-Stock Direction Predictor")
st.caption(f"Data from **{start}** to **{end}** · Model: Random Forest (300 trees)")

# ── Top metrics row ─────────────────────────────────────────────────────────
last_close = float(df["Close"].iloc[-1])
prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else last_close
change = last_close - prev_close
change_pct = (change / prev_close) * 100 if prev_close else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Stock", stock)
col2.metric("Last Close", f"${last_close:,.2f}", f"{change_pct:+.2f}%")
col3.metric(
    "Prediction",
    "UP 📈" if prediction == 1 else "DOWN 📉",
)
col4.metric("Confidence", f"{confidence:.1f}%")

st.divider()

# ── Two-column layout: Chart + Indicators ────────────────────────────────────
left, right = st.columns([3, 1])

with left:
    st.subheader("Price & Moving Averages")

    fig = go.Figure()

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC",
        )
    )

    # SMA overlays
    fig.add_trace(
        go.Scatter(x=df["Date"], y=df["SMA_10"], name="SMA 10",
                   line=dict(width=1.2, color="#2196f3"))
    )
    fig.add_trace(
        go.Scatter(x=df["Date"], y=df["SMA_20"], name="SMA 20",
                   line=dict(width=1.2, color="#ff9800"))
    )

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=460,
        margin=dict(l=0, r=0, t=10, b=0),
    )

    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Latest Indicators")

    latest_row = df.iloc[-1]
    st.metric("RSI (14)", f"{latest_row['RSI']:.2f}")
    st.metric("SMA 10", f"{latest_row['SMA_10']:.2f}")
    st.metric("SMA 20", f"{latest_row['SMA_20']:.2f}")
    st.metric("Volatility (10d)", f"{latest_row['Volatility']:.4f}")
    st.metric("Daily Return", f"{latest_row['Return'] * 100:.2f}%")

st.divider()

# ── Volume chart ─────────────────────────────────────────────────────────────
st.subheader("Volume")

vol_colors = np.where(df["Close"] >= df["Open"], "#00c853", "#ff1744")

vol_fig = go.Figure(
    go.Bar(x=df["Date"], y=df["Volume"], marker_color=vol_colors, name="Volume")
)
vol_fig.update_layout(
    template="plotly_dark",
    height=250,
    margin=dict(l=0, r=0, t=10, b=0),
)
st.plotly_chart(vol_fig, use_container_width=True)

# ── RSI chart ────────────────────────────────────────────────────────────────
st.subheader("RSI (14)")

rsi_fig = go.Figure()
rsi_fig.add_trace(
    go.Scatter(x=df["Date"], y=df["RSI"], name="RSI",
               line=dict(color="#ab47bc", width=1.5))
)
rsi_fig.add_hline(y=70, line_dash="dash", line_color="red",   annotation_text="Overbought")
rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
rsi_fig.update_layout(
    template="plotly_dark",
    height=250,
    yaxis=dict(range=[0, 100]),
    margin=dict(l=0, r=0, t=10, b=0),
)
st.plotly_chart(rsi_fig, use_container_width=True)

# ── Raw data expander ───────────────────────────────────────────────────────
with st.expander("📋 View Raw Data"):
    st.dataframe(
        df[["Date", "Open", "High", "Low", "Close", "Volume",
            "Return", "SMA_10", "SMA_20", "Volatility", "RSI"]].tail(60),
        use_container_width=True,
    )

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("⚠️ This is an educational tool, not financial advice. Past performance does not predict future results.")
 