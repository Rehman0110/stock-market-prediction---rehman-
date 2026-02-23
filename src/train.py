import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

from src.data_loader import download_data
from src.features import add_indicators
from src.preprocess import encode_stock, get_features
from src.config import MODEL_PATH


def train():

    df = download_data()
    df = add_indicators(df)
    df = encode_stock(df)

    X, y = get_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print(classification_report(y_test, preds))

    joblib.dump(model, MODEL_PATH)

    print("Model saved successfully.")


if __name__ == "__main__":
    train()