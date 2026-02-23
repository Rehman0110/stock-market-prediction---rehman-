import yfinance as yf
import pandas as pd
from src.config import STOCKS, START_DATE, END_DATE


def download_data():
    frames = []

    for stock in STOCKS:
        df = yf.download(stock, start=START_DATE, end=END_DATE)

        # Flatten multi-level columns from newer yfinance versions
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df["Stock"] = stock
        df.reset_index(inplace=True)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)