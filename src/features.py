# src/features.py

import numpy as np


def add_indicators(df):

    df = df.copy()

    # Sort properly
    df = df.sort_values(["Stock", "Date"])

    # Group by stock
    grouped = df.groupby("Stock")

    df["Return"] = grouped["Close"].pct_change(fill_method=None)

    df["SMA_10"] = grouped["Close"].transform(lambda x: x.rolling(10).mean())
    df["SMA_20"] = grouped["Close"].transform(lambda x: x.rolling(20).mean())

    df["Volatility"] = grouped["Return"].transform(lambda x: x.rolling(10).std())

    # RSI
    delta = grouped["Close"].diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    df["_gain"] = gain
    df["_loss"] = loss

    avg_gain = df.groupby("Stock")["_gain"].transform(lambda x: x.rolling(14).mean())
    avg_loss = df.groupby("Stock")["_loss"].transform(lambda x: x.rolling(14).mean())

    rs = avg_gain / avg_loss

    df["RSI"] = 100 - (100 / (1 + rs))

    df.drop(columns=["_gain", "_loss"], inplace=True)

    # Target
    df["Target"] = grouped["Close"].shift(-1) > df["Close"]
    df["Target"] = df["Target"].astype(int)

    df.dropna(inplace=True)

    return df