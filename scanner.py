import yfinance as yf
import pandas as pd
import ta
import requests
from xgboost import XGBClassifier
from sklearn.utils import resample
import os

# ==============================
# 🔹 STEP 0: CONFIG
# ==============================
capital = 100000
max_per_stock = 0.4

BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

# ==============================
# 🔹 STEP 1: STOCK LIST (AUTO UNIVERSE)
# ==============================
stocks = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
    "SBIN.NS","AXISBANK.NS","KOTAKBANK.NS","ITC.NS","LT.NS",
    "BHARTIARTL.NS","ASIANPAINT.NS","MARUTI.NS","SUNPHARMA.NS",
    "TITAN.NS","ULTRACEMCO.NS","WIPRO.NS","ONGC.NS","NTPC.NS",
    "POWERGRID.NS","ADANIENT.NS","ADANIPORTS.NS","BAJFINANCE.NS",
    "BAJAJFINSV.NS","HCLTECH.NS","TECHM.NS","NESTLEIND.NS",
    "INDUSINDBK.NS","COALINDIA.NS","TATASTEEL.NS","JSWSTEEL.NS"
]

results = []

# ==============================
# 🔹 STEP 2: MARKET FILTER
# ==============================
nifty = yf.download("^NSEI", start="2023-01-01")
nifty.columns = nifty.columns.get_level_values(0)

nifty_close = nifty['Close'].squeeze()
nifty['MA50'] = nifty_close.rolling(50).mean()

market_uptrend = nifty_close.iloc[-1] > nifty['MA50'].iloc[-1]

# ==============================
# 🔹 FUNCTIONS
# ==============================
def backtest_strategy(data):
    position = 0
    positions = []

    for i in range(len(data)):
        if position == 0 and data['Entry'].iloc[i]:
            position = 1
        elif position == 1 and data['Exit'].iloc[i]:
            position = 0
        positions.append(position)

    data['Position'] = positions
    data['Market_Return'] = data['Close'].pct_change()
    data['Strategy_Return'] = data['Position'].shift(1) * data['Market_Return']

    data['Cumulative_Strategy'] = (1 + data['Strategy_Return']).cumprod()
    data['Cumulative_Market'] = (1 + data['Market_Return']).cumprod()

    return data


def calculate_metrics(bt):
    bt = bt.dropna()

    returns = bt['Strategy_Return']

    total_return = bt['Cumulative_Strategy'].iloc[-1] - 1
    win_rate = (returns > 0).sum() / len(returns)

    peak = bt['Cumulative_Strategy'].cummax()
    drawdown = (bt['Cumulative_Strategy'] - peak) / peak
    max_dd = drawdown.min()

    sharpe = returns.mean() / returns.std() if returns.std() != 0 else 0

    return total_return, win_rate, max_dd, sharpe

# ==============================
# 🔹 STEP 3: STOCK LOOP
# ==============================
for stock in stocks:
    print(f"Processing {stock}...")

    try:
        data = yf.download(stock, start="2020-01-01")
        data.columns = data.columns.get_level_values(0)

        if data.empty:
            continue

        close = data['Close'].squeeze()

        # ==============================
        # 🔹 STEP 3A: PRE-FILTER
        # ==============================
        avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
        price = close.iloc[-1]

        if avg_volume < 1000000:
            continue

        if price < 50 or price > 2000:
            continue

        # Indicators
        data['RSI'] = ta.momentum.RSIIndicator(close=close).rsi()
        data['MA20'] = close.rolling(20).mean()
        data['MA50'] = close.rolling(50).mean()

        macd = ta.trend.MACD(close)
        data['MACD'] = macd.macd()
        data['MACD_signal'] = macd.macd_signal()

        # ==============================
        # 🔹 STEP 3B: MOMENTUM FILTER
        # ==============================
        if not (close.iloc[-1] > data['MA20'].iloc[-1] and data['RSI'].iloc[-1] > 50):
            continue

        # Entry / Exit
        data['Entry'] = (
            (data['RSI'] > 55) &
            (data['MA20'] > data['MA50']) &
            (data['MACD'] > data['MACD_signal'])
        )

        data['Exit'] = (
            (data['RSI'] < 45) |
            (data['MACD'] < data['MACD_signal'])
        )

        # ATR
        atr = ta.volatility.AverageTrueRange(
            high=data['High'],
            low=data['Low'],
            close=close
        )
        data['ATR'] = atr.average_true_range()

        # Target
        data['Future_Return'] = close.shift(-5) / close - 1
        data['Target'] = (data['Future_Return'] > 0.03).astype(int)

        data = data.dropna()

        # Backtest
        bt = backtest_strategy(data)
        strategy_return, win_rate, max_dd, sharpe = calculate_metrics(bt)

        # ML
        features = ['RSI','MACD','MACD_signal','MA20','MA50','ATR','Volume']

        df_majority = data[data['Target'] == 0]
        df_minority = data[data['Target'] == 1]

        if len(df_minority) == 0:
            continue

        df_minority = resample(df_minority, replace=True, n_samples=len(df_majority))
        data_balanced = pd.concat([df_majority, df_minority])

        X = data_balanced[features]
        y = data_balanced['Target']

        model = XGBClassifier(n_estimators=200, max_depth=4)
        model.fit(X, y)

        prob = model.predict_proba(X.iloc[-1:])[:, 1][0]

        if prob < 0.55:
            continue

        results.append({
            "Stock": stock,
            "Probability": prob,
            "ATR": data.iloc[-1]['ATR'],
            "Price": price,
            "Strategy_Return": strategy_return,
            "WinRate": win_rate,
            "Sharpe": sharpe
        })

    except Exception as e:
        print(f"Error in {stock}: {e}")

# ==============================
# 🔹 STEP 4: DATAFRAME + SCORING
# ==============================
df = pd.DataFrame(results)

message = "📊 AI STOCK SIGNALS\n\n"
message += "📈 MARKET: UPTREND ✅\n\n" if market_uptrend else "📉 MARKET: DOWNTREND ❌\n\n"

portfolio = []

if not df.empty:

    df['Score'] = (
        df['Probability'] * 0.5 +
        df['Sharpe'] * 0.3 +
        df['Strategy_Return'] * 0.2
    ) / df['Price']

    df = df.sort_values(by="Score", ascending=False)

    # ==============================
    # 🔹 STEP 5: AUTO SELECTION
    # ==============================
    top_stocks = df.head(5)

    # ==============================
    # 🔹 STEP 6: PORTFOLIO
    # ==============================
    total_prob = top_stocks['Probability'].sum()

    for _, row in top_stocks.iterrows():
        allocation = min(capital * (row['Probability']/total_prob), capital*max_per_stock)

        sl = row['Price'] - (row['ATR'] * 2)
        target = row['Price'] + (row['ATR'] * 4)

        portfolio.append({
            "Stock": row['Stock'],
            "Entry": round(row['Price'],2),
            "SL": round(sl,2),
            "Target": round(target,2),
            "Allocation": round(allocation)
        })

# ==============================
# 🔹 STEP 7: MESSAGE
# ==============================
message += "💼 PORTFOLIO:\n"

if portfolio:
    for p in portfolio:
        message += f"{p['Stock']} | Entry ₹{p['Entry']} | SL ₹{p['SL']} | Target ₹{p['Target']}\n"
else:
    message += "No trades today\n"

# ==============================
# 🔹 STEP 8: TELEGRAM
# ==============================
response = requests.post(
    f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
    data={"chat_id": CHAT_ID, "text": message}
)

print("Telegram response:", response.text)
