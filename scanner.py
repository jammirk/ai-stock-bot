import yfinance as yf
import pandas as pd
import numpy as np
import ta
from xgboost import XGBClassifier

# List of stocks (you can expand later)
stocks = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS"
]

results = []

for stock in stocks:
    print(f"Processing {stock}...")

    try:
        # Download data
        data = yf.download(stock, start="2020-01-01", end="2025-01-01")
        data.columns = data.columns.get_level_values(0)

        close = data['Close'].squeeze()

        # Indicators
        data['RSI'] = ta.momentum.RSIIndicator(close=close).rsi()
        data['MA20'] = close.rolling(20).mean()
        data['MA50'] = close.rolling(50).mean()
        data['Trend'] = data['MA20'] > data['MA50']

        macd = ta.trend.MACD(close)
        data['MACD'] = macd.macd()
        data['MACD_signal'] = macd.macd_signal()

        data['Returns'] = close.pct_change()

        # Target
        data['Future_Return'] = close.shift(-5) / close - 1
        data['Target'] = (data['Future_Return'] > 0.02).astype(int)

        data = data.dropna()

        features = ['RSI', 'MA20', 'MA50', 'MACD', 'MACD_signal', 'Returns']
        X = data[features]
        y = data['Target']

        split = int(len(X) * 0.8)

        X_train = X[:split]
        y_train = y[:split]

        # Train model
        model = XGBClassifier(n_estimators=100, max_depth=4)
        model.fit(X_train, y_train)

        # Latest prediction
        latest = data.iloc[-1]
        prob = model.predict_proba(X.iloc[-1:])[:, 1][0]
        trend = latest['Trend']

        results.append({
            "Stock": stock,
            "Probability": prob,
            "Trend": trend
        })

    except Exception as e:
        print(f"Error in {stock}: {e}")

# Convert to DataFrame
df = pd.DataFrame(results)

# Sort
df = df.sort_values(by="Probability", ascending=False)

# Apply filters (WITH TREND)
buy_signals = df[(df['Probability'] > 0.6) & (df['Trend'] == True)]
sell_signals = df[(df['Probability'] < 0.4) & (df['Trend'] == False)]

print("\n🔥 STRONG BUY SIGNALS:\n")
print(buy_signals)

print("\n⚠️ STRONG SELL SIGNALS:\n")
print(sell_signals)