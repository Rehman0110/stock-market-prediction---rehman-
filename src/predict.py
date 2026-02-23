import joblib
from src.config import MODEL_PATH, ENCODER_PATH


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, encoder