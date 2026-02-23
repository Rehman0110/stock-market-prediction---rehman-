# 📈 Multi-Stock Direction Predictor

🌐 **Live Demo:** [stock-market-prediction-rehman.onrender.com](https://stock-market-prediction-rehman.onrender.com)

A machine learning application that predicts whether a stock's price will go **UP** or **DOWN** the next trading day, using a Random Forest classifier trained on historical market data and technical indicators.

## Stocks Covered

| Ticker        | Company              |
|---------------|----------------------|
| AAPL          | Apple                |
| MSFT          | Microsoft            |
| TSLA          | Tesla                |
| RELIANCE.NS   | Reliance Industries  |
| TCS.NS        | Tata Consultancy     |

## Project Structure

```
myenv/
├── app/
│   └── app.py              # Streamlit dashboard
├── src/
│   ├── config.py            # Stocks list, date range, file paths
│   ├── data_loader.py       # Downloads historical data via yfinance
│   ├── features.py          # Technical indicators (SMA, RSI, Volatility)
│   ├── preprocess.py        # Label encoding & feature selection
│   ├── train.py             # Model training script
│   └── predict.py           # Model & encoder loading utilities
├── models/
│   ├── rf_model.pkl         # Trained Random Forest model
│   └── label_encoder.pkl    # Fitted label encoder
├── requirements.txt
└── README.md
```

## Features

### Technical Indicators
- **SMA 10 / SMA 20** — Simple Moving Averages (10-day & 20-day)
- **RSI (14)** — Relative Strength Index
- **Volatility** — 10-day rolling standard deviation of returns
- **Daily Return** — Percentage change in closing price

### Model
- **Algorithm:** Random Forest Classifier
- **Trees:** 300 estimators
- **Max Depth:** 12
- **Training Data:** 2015-01-01 to 2024-12-31 (80/20 time-series split)

### Dashboard (Streamlit)
- Stock selector & date range picker
- UP/DOWN prediction with confidence score
- Interactive candlestick chart with SMA overlays
- Volume bar chart (green/red)
- RSI chart with overbought/oversold thresholds
- Latest indicator values panel
- Raw data viewer

## Setup

### 1. Create & activate virtual environment

```bash
python3 -m venv myenv
source myenv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
cd myenv
python src/train.py
```

This downloads historical data, computes features, trains the Random Forest, and saves the model to `models/rf_model.pkl`.

### 4. Launch the dashboard

```bash
streamlit run app/app.py
```

Open **http://localhost:8501** in your browser.

## Requirements

- Python 3.10+
- pandas
- numpy
- scikit-learn
- yfinance
- joblib
- streamlit
- matplotlib
- plotly

## Disclaimer

⚠️ This is an **educational project** and not financial advice. Past performance does not guarantee future results. Do not use this tool for real trading decisions.
