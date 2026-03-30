import yfinance as yf
import pandas as pd
import numpy as np
import ta
import requests
from xgboost import XGBClassifier

# ==============================
# 🔹 STEP 0: CONFIG
# ==============================
capital = 100000
max_per_stock = 0.4
risk_per_trade = 0.02
stop_loss_pct = 0.05

import os

BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

# ==============================
# 🔹 STEP 1: STOCK LIST
# ==============================
stocks = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
    "HINDUNILVR.NS","ITC.NS","LT.NS","SBIN.NS","AXISBANK.NS",
    "KOTAKBANK.NS","BHARTIARTL.NS","ASIANPAINT.NS","MARUTI.NS",
    "SUNPHARMA.NS","TITAN.NS","ULTRACEMCO.NS","WIPRO.NS",
    "ONGC.NS","NTPC.NS"
]

results = []

# ==============================
# 🔹 STEP 2: MARKET FILTER (NIFTY)
# ==============================
nifty = yf.download("^NSEI", start="2020-01-01", end="2025-01-01")
nifty.columns = nifty.columns.get_level_values(0)

nifty_close = nifty['Close'].squeeze()
nifty['MA50'] = nifty_close.rolling(50).mean()

market_uptrend = nifty_close.iloc[-1] > nifty['MA50'].iloc[-1]

# ==============================
# 🔹 STEP 3: STOCK LOOP
# ==============================
for stock in stocks:
    print(f"Processing {stock}...")

    try:
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
        
        # 🔥 ATR (volatility)
        atr = ta.volatility.AverageTrueRange(
            high=data['High'],
            low=data['Low'],
            close=close
        )
        data['ATR'] = atr.average_true_range()

        # Target
        data['Future_Return'] = close.shift(-5) / close - 1
        data['Target'] = (data['Future_Return'] > 0.02).astype(int)

        data = data.dropna()

        features = ['RSI','MA20','MA50','MACD','MACD_signal','Returns']
        X = data[features]
        y = data['Target']

        split = int(len(X) * 0.8)

        X_train = X[:split]
        y_train = y[:split]

        model = XGBClassifier(n_estimators=100, max_depth=4)
        model.fit(X_train, y_train)

        prob = model.predict_proba(X.iloc[-1:])[:, 1][0]
        trend = data.iloc[-1]['Trend']
        atr_value = data.iloc[-1]['ATR']
        price = close.iloc[-1]

        results.append({
            "Stock": stock,
            "Probability": prob,
            "Trend": trend,
            "ATR": atr_value,
            "Price": price
        })

    except Exception as e:
        print(f"Error in {stock}: {e}")

# ==============================
# 🔹 STEP 4: DATAFRAME
# ==============================
df = pd.DataFrame(results)
df['Score'] = df['Probability'] / df['Price'].replace(0, 1)

# ==============================
# 🔹 ALWAYS PREPARE MESSAGE
# ==============================
message = "📊 AI STOCK SIGNALS\n\n"
message += "📈 MARKET: UPTREND ✅\n\n" if market_uptrend else "📉 MARKET: DOWNTREND ❌\n\n"

if df.empty:
    print("❌ No data available")
    message += "⚠️ No data available today\n"

else:
    df = df.sort_values(by="Probability", ascending=False)

    print("\n📊 ALL STOCK SCORES:\n")
    print(df)

    print("\n📈 MARKET TREND:", "UPTREND ✅" if market_uptrend else "DOWNTREND ❌")

    # ==============================
    # 🔹 STEP 5: PORTFOLIO SELECTION (FIXED)
    # ==============================
if market_uptrend:
    top_stocks = df[
        (df['Probability'] > 0.6) & 
        (df['Trend'] == True)
    ].sort_values(by="Score", ascending=False).head(3)
else:
    top_stocks = pd.DataFrame()

    # ==============================
    # 🔹 STEP 6: PORTFOLIO ALLOCATION
    # ==============================
    portfolio = []

    if not top_stocks.empty:
        total_prob = top_stocks['Probability'].sum()

        for _, row in top_stocks.iterrows():
            weight = row['Probability'] / total_prob
            allocation = capital * weight
            allocation = min(allocation, capital * max_per_stock)

            stop_loss_price = row['Price'] - (row['ATR'] * 2)
            stop_loss_pct_dynamic = ((row['Price'] - stop_loss_price) / row['Price']) * 100

            portfolio.append({
                "Stock": row['Stock'],
                "Probability": row['Probability'],
                "Allocation": round(allocation),
                "StopLoss": f"{round(stop_loss_pct_dynamic,1)}%",
                "SL_Price": round(stop_loss_price, 2)
            })

    # ==============================
    # 🔹 STEP 7: PRINT PORTFOLIO
    # ==============================
    print("\n💼 AI PORTFOLIO:\n")

    if portfolio:
        for p in portfolio:
            print(f"{p['Stock']} → ₹{p['Allocation']} | SL: {p['StopLoss']} @ {p['SL_Price']}")
    else:
        print("No trades due to market condition")

    # ==============================
    # 🔹 STEP 8: BUY / SELL SIGNALS
    # ==============================
    buy_signals = df[(df['Probability'] > 0.6) & (df['Trend'] == True)].head(3)
    sell_signals = df[(df['Probability'] < 0.4) & (df['Trend'] == False)].head(3)

    print("\n🔥 STRONG BUY SIGNALS:\n", buy_signals)
    print("\n⚠️ STRONG SELL SIGNALS:\n", sell_signals)

    # ==============================
    # 🔹 STEP 9: TELEGRAM MESSAGE
    # ==============================
    message += "💼 AI PORTFOLIO:\n"

    if portfolio:
        for p in portfolio:
            message += f"{p['Stock']} → ₹{p['Allocation']} | SL: {p['StopLoss']} @ {p['SL_Price']}\n"
    else:
        message += "No trades due to market condition\n"

    message += "\n🔥 BUY SIGNALS:\n"
    if not buy_signals.empty:
        for _, row in buy_signals.iterrows():
            label = "HIGH" if row['Probability'] > 0.7 else "MEDIUM"
            message += f"{row['Stock']} - {round(row['Probability'],2)} ({label})\n"
    else:
        message += "No strong buys\n"

    message += "\n⚠️ SELL SIGNALS:\n"
    if not sell_signals.empty:
        for _, row in sell_signals.iterrows():
            label = "HIGH" if row['Probability'] > 0.7 else "MEDIUM"
            message += f"{row['Stock']} - {round(row['Probability'],2)} ({label})\n"
    else:
        message += "No strong sells\n"

# ==============================
# 🔹 ALWAYS SEND TELEGRAM
# ==============================
response = requests.post(
    f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
    data={"chat_id": CHAT_ID, "text": message}
)

print("Telegram response:", response.text)
