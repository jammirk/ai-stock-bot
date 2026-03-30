import yfinance as yf
import pandas as pd
import numpy as np
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
# 🔹 STEP 2: MARKET FILTER
# ==============================
nifty = yf.download("^NSEI", start="2020-01-01", end="2025-01-01")
nifty.columns = nifty.columns.get_level_values(0)

nifty_close = nifty['Close'].squeeze()
nifty['MA50'] = nifty_close.rolling(50).mean()

market_uptrend = nifty_close.iloc[-1] > nifty['MA50'].iloc[-1]

def backtest_strategy(data):
    data = data.copy()

    position = 0
    positions = []

    for i in range(len(data)):
        # BUY
        if position == 0 and data['Entry'].iloc[i]:
            position = 1

        # SELL
        elif position == 1 and data['Exit'].iloc[i]:
            position = 0

        positions.append(position)

    data['Position'] = positions

    # Returns
    data['Market_Return'] = data['Close'].pct_change()
    data['Strategy_Return'] = data['Position'].shift(1) * data['Market_Return']

    # Cumulative performance
    data['Cumulative_Strategy'] = (1 + data['Strategy_Return']).cumprod()
    data['Cumulative_Market'] = (1 + data['Market_Return']).cumprod()

    return data
    
    def calculate_metrics(bt):
        bt = bt.dropna()
    
        returns = bt['Strategy_Return']
    
        total_return = bt['Cumulative_Strategy'].iloc[-1] - 1
    
        win_rate = (returns > 0).sum() / len(returns)
    
        # Max Drawdown
        cumulative = bt['Cumulative_Strategy']
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()
    
        # Sharpe Ratio
        sharpe = returns.mean() / returns.std() if returns.std() != 0 else 0
    
        return total_return, win_rate, max_drawdown, sharpe

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

        # 🔥 ENTRY CONDITIONS
        data['Entry'] = (
            (data['RSI'] > 55) &
            (data['MA20'] > data['MA50']) &
            (data['MACD'] > data['MACD_signal']) &
            (close > data['MA20'])  # breakout strength
        )

        # 🔥 EXIT CONDITIONS
        data['Exit'] = (
            (data['RSI'] < 45) |
            (data['MACD'] < data['MACD_signal']) |
            (close < data['MA20'])
        )

        # Returns
        data['Returns'] = close.pct_change()
        data['Return_3'] = close.pct_change(3)
        data['Return_5'] = close.pct_change(5)

        # Momentum filter
        data['Momentum'] = (
            (close > data['MA20']) &
            (data['RSI'] > 55) &
            (data['Returns'] > 0)
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
        
        # 🔥 BACKTEST
        bt = backtest_strategy(data)
        strategy_return, win_rate, max_dd, sharpe = calculate_metrics(bt)
        
        # Final performance
        strategy_return = bt['Cumulative_Strategy'].iloc[-1]
        market_return = bt['Cumulative_Market'].iloc[-1]

        # Features
        features = [
            'RSI','MACD','MACD_signal','Returns',
            'Return_3','Return_5','ATR','MA20','MA50','Volume'
        ]

        # ==============================
        # 🔹 STEP 5: BALANCE DATA
        # ==============================
        df_majority = data[data['Target'] == 0]
        df_minority = data[data['Target'] == 1]

        if len(df_minority) == 0:
            continue

        df_minority_upsampled = resample(
            df_minority,
            replace=True,
            n_samples=len(df_majority),
            random_state=42
        )

        data_balanced = pd.concat([df_majority, df_minority_upsampled])

        X = data_balanced[features]
        y = data_balanced['Target']

        # ==============================
        # 🔹 STEP 6: TRAIN MODEL
        # ==============================
        split = int(len(X) * 0.8)

        X_train = X.iloc[:split]
        y_train = y.iloc[:split]

        model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        model.fit(X_train, y_train)

        # ==============================
        # 🔹 STEP 7: PREDICTION
        # ==============================
        prob = model.predict_proba(X.iloc[-1:])[:, 1][0]

        if prob < 0.55:
            continue

        results.append({
            "Stock": stock,
            "Probability": prob,
            "Trend": data.iloc[-1]['Trend'],
            "Momentum": data.iloc[-1]['Momentum'],
            "ATR": data.iloc[-1]['ATR'],
            "Price": close.iloc[-1],
            "Strategy_Return": strategy_return,
            "Market_Return": market_return,
            "Strategy_Return": strategy_return,
            "WinRate": win_rate,
            "MaxDrawdown": max_dd,
            "Sharpe": sharpe,
            })

    except Exception as e:
        print(f"Error in {stock}: {e}")

# ==============================
# 🔹 STEP 4: DATAFRAME
# ==============================
df = pd.DataFrame(results)

message = "📊 AI STOCK SIGNALS\n\n"
message += "📈 MARKET: UPTREND ✅\n\n" if market_uptrend else "📉 MARKET: DOWNTREND ❌\n\n"

if df.empty:
    print("❌ No data available")
    message += "No signals today\n"

else:
    df['Score'] = df['Probability'] / df['Price'].replace(0, 1)
    df = df.sort_values(by="Score", ascending=False)

    print("\n📊 ALL STOCK SCORES:\n", df)
    
    # ==============================
    # 🔹 WATCHLIST (TOP OPPORTUNITIES)
    # ==============================
    watchlist = df.head(5)

    print("\n👀 WATCHLIST (NEAR SIGNALS):\n")
    print(watchlist[['Stock','Probability','Score']])

    print("\n📊 BACKTEST RESULTS:\n")
    print(df[['Stock','Strategy_Return','Market_Return']])

    print("\n📊 PERFORMANCE METRICS:\n")
    print(df[['Stock','Strategy_Return','WinRate','MaxDrawdown','Sharpe']])

    # ==============================
    # 🔹 STEP 5: STOCK SELECTION
    # ==============================
    if market_uptrend:
        filtered_df = df[
            (df['Probability'] > 0.6) &
            (df['Trend'] == True) &
            (df['Momentum'] == True) &
            (df['Strategy_Return'] > 0.2) &   # 🔥 profitable
            (df['WinRate'] > 0.5) &           # 🔥 consistency
            (df['Sharpe'] > 0.5) &            # 🔥 risk-adjusted return
            (df['Price'] > 100) &
            (df['Price'] < 1500)
        ]
        top_stocks = filtered_df.head(3)
    else:
        top_stocks = pd.DataFrame()

    # ==============================
    # 🔹 STEP 6: PORTFOLIO
    # ==============================
    portfolio = []

    if not top_stocks.empty:
        total_prob = top_stocks['Probability'].sum()

        for _, row in top_stocks.iterrows():
            weight = row['Probability'] / total_prob
            allocation = min(capital * weight, capital * max_per_stock)

            # 🔥 Dynamic Stop Loss
            sl_price = row['Price'] - (row['ATR'] * 2)
            
            # 🔥 Target (Risk:Reward 1:2)
            target_price = row['Price'] + (row['ATR'] * 4)

            portfolio.append({
                "Stock": row['Stock'],
                "Allocation": round(allocation),
                "Entry": round(row['Price'], 2),
                "SL": round(sl_price, 2),
                "Target": round(target_price, 2)
            })

    # ==============================
    # 🔹 STEP 7: MESSAGE
    # ==============================
    message += "💼 PORTFOLIO:\n"

if portfolio:
    for p in portfolio:
        message += (
            f"{p['Stock']}\n"
            f"Entry: ₹{p['Entry']}\n"
            f"SL: ₹{p['SL']}\n"
            f"Target: ₹{p['Target']}\n\n"
        )
else:
    message += "No trades today\n"
    
# ==============================
# 🔹 TELEGRAM SEND
# ==============================
response = requests.post(
    f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
    data={"chat_id": CHAT_ID, "text": message}
)

print("Telegram response:", response.text)
