from sklearn.preprocessing import LabelEncoder
import joblib
from src.config import ENCODER_PATH


def encode_stock(df):

    le = LabelEncoder()
    df["Stock"] = le.fit_transform(df["Stock"])

    joblib.dump(le, ENCODER_PATH)

    return df


def get_features(df):

    feature_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "Return", "SMA_10", "SMA_20", "Volatility", "RSI", "Stock"
    ]

    X = df[feature_cols]
    y = df["Target"]

    return X, y