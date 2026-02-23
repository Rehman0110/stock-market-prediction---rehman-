import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STOCKS = [
    "AAPL",
    "MSFT",
    "TSLA",
    "RELIANCE.NS",
    "TCS.NS"
]

START_DATE = "2015-01-01"
END_DATE = "2024-12-31"

MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "rf_model.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")