import yfinance as yf
import pandas as pd
import ta
import requests
from xgboost import XGBClassifier
from sklearn.utils import resample
import os
from dotenv import load_dotenv
from datetime import datetime, timezone

# ==============================
# 🔹 STEP 0: CONFIG
# ==============================
load_dotenv()

capital = 100000
max_per_stock = 0.4

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# ==============================
# 🔹 STEP 1: MARKET PHASE
# ==============================
def get_market_phase():
    now = datetime.now(timezone.utc)
    hour = now.hour + 5.5

    if 9 <= hour < 15.5:
        return "LIVE"
    elif 8 <= hour < 9:
        return "PRE"
    else:
        return "CLOSED"

market_phase = get_market_phase()
print("Market Phase:", market_phase)

# ==============================
# 🔹 STEP 2: STOCK LIST
# ==============================
stocks = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
    "SBIN.NS","AXISBANK.NS","KOTAKBANK.NS","ITC.NS","LT.NS",
    "BHARTIARTL.NS","ASIANPAINT.NS","MARUTI.NS","SUNPHARMA.NS",
    "TITAN.NS","WIPRO.NS","ONGC.NS","NTPC.NS",
    "POWERGRID.NS","ADANIENT.NS","ADANIPORTS.NS",
    "HCLTECH.NS","TECHM.NS","INDUSINDBK.NS","COALINDIA.NS",
    "TATASTEEL.NS","JSWSTEEL.NS"
]

results = []

# ==============================
# 🔹 STEP 3: MARKET TREND
# ==============================
nifty = yf.download("^NSEI", period="1y")

if isinstance(nifty.columns, pd.MultiIndex):
    nifty.columns = nifty.columns.get_level_values(0)

nifty_close = nifty['Close'].squeeze()
nifty_ma50 = nifty_close.rolling(50).mean()

last_close = float(nifty_close.iloc[-1])
last_ma50 = float(nifty_ma50.iloc[-1]) if not pd.isna(nifty_ma50.iloc[-1]) else None

market_uptrend = False if last_ma50 is None else last_close > last_ma50

# ==============================
# 🔹 FUNCTIONS
# ==============================
def get_live_price(symbol):
    try:
        data = yf.download(symbol, period="1d", interval="1m")
        if not data.empty:
            return float(data['Close'].iloc[-1])
    except:
        pass
    return None


def send_signal(stock, entry, sl, target):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    msg = (
        f"📊 AI SIGNAL\n\n"
        f"{stock}\n"
        f"Entry: ₹{entry}\n"
        f"SL: ₹{sl}\n"
        f"Target: ₹{target}"
    )

    requests.post(url, data={"chat_id": CHAT_ID, "text": msg})


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
# 🔹 STEP 4: STOCK LOOP
# ==============================
for stock in stocks:
    print(f"Processing {stock}...")

    try:
        data = yf.download(stock, period="2y")

        if data.empty:
            continue

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        close = data['Close']

        price = close.iloc[-1]
        avg_volume = data['Volume'].rolling(20).mean().iloc[-1]

        if price < 50 or price > 2000 or avg_volume < 500000:
            continue

        data['RSI'] = ta.momentum.RSIIndicator(close).rsi()
        data['MA20'] = close.rolling(20).mean()
        data['MA50'] = close.rolling(50).mean()
        # 🔹 Trend strength filter
        trend_strength = (data['MA20'].iloc[-1] - data['MA50'].iloc[-1]) / data['MA50'].iloc[-1]

        if trend_strength < 0.01:
            continue
        data['Vol_Avg'] = data['Volume'].rolling(20).mean()

        # Volume breakout condition
        #if data['Volume'].iloc[-1] < data['Vol_Avg'].iloc[-1]:
            #continue

        macd = ta.trend.MACD(close)
        data['MACD'] = macd.macd()
        data['MACD_signal'] = macd.macd_signal()

        #  Additional ML Features
        data['Returns'] = close.pct_change()
        data['Volatility'] = data['Returns'].rolling(10).std()
        data['Trend'] = data['MA20'] - data['MA50']
        # 🔹 Relative Strength vs Nifty
        nifty_recent = nifty_close.pct_change(20).iloc[-1]
        stock_recent = close.pct_change(20).iloc[-1]

        if stock_recent < nifty_recent:
            continue

        if not (close.iloc[-1] > data['MA20'].iloc[-1] and data['RSI'].iloc[-1] > 50):
            continue

        data['Entry'] = (
        (data['RSI'] > 55) &
        (data['RSI'] < 75) &   # avoid overbought
        (data['MA20'] > data['MA50']) &
        (data['MACD'] > data['MACD_signal'])
)
        data['Exit'] = (data['RSI'] < 45) | (data['MACD'] < data['MACD_signal'])

        atr = ta.volatility.AverageTrueRange(data['High'], data['Low'], close)
        data['ATR'] = atr.average_true_range()

        data['Future_Return'] = close.shift(-5) / close - 1
        data['Target'] = ((data['Future_Return'] > 0.025) & (data['RSI'] > 50)).astype(int)

        data = data.dropna()

        bt = backtest_strategy(data)
        strategy_return, win_rate, max_dd, sharpe = calculate_metrics(bt)

        features = ['RSI','MACD','MACD_signal','MA20','MA50','ATR','Volume','Volatility','Trend']

        df_major = data[data['Target'] == 0]
        df_minor = data[data['Target'] == 1]

        if len(df_minor) == 0:
            continue

        df_minor = resample(df_minor, replace=True, n_samples=len(df_major))
        balanced = pd.concat([df_major, df_minor])

        X = balanced[features]
        y = balanced['Target']

        model = XGBClassifier(n_estimators=200, max_depth=4)
        split = int(len(X) * 0.8)

        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        model.fit(X_train, y_train)

        prob = model.predict_proba(X.iloc[-1:])[:, 1][0]
        # 🔹 Market adjustment (instead of blocking)
        if not market_uptrend:
            prob = prob * 0.75

        if prob < 0.55:
            continue
        
        risk = data.iloc[-1]['ATR'] * 2
        reward = data.iloc[-1]['ATR'] * 4

        if reward / risk < 2:
            continue
        
        print(f"✅ Adding {stock} to results")

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
# 🔹 STEP 5: DATAFRAME
# ==============================
print("Total selected stocks:", len(results))
if len(results) == 0:
    print("⚠️ No ML signals, using fallback")

    for stock in stocks[:5]:
        results.append({
            "Stock": stock,
            "Probability": 0.5,
            "ATR": 10,
            "Price": 100,
            "Strategy_Return": 0.01,
            "WinRate": 0.5,
            "Sharpe": 0.5
        })
df = pd.DataFrame(results)
portfolio = []

# ==============================
# 🔹 STEP 6: SELECTION
# ==============================
if not df.empty:

    df['Score'] = (
        df['Probability']*0.4 +
        df['Sharpe']*0.3 +
        df['Strategy_Return']*0.2 +
        df['WinRate']*0.1
    )
    
    df = df.sort_values(by="Score", ascending=False)

    top_stocks = df.head(5)

    total_prob = top_stocks['Probability'].sum()

    for _, row in top_stocks.iterrows():

        entry = row['Price']
        atr = row['ATR']

        allocation = min(capital * max_per_stock, capital / len(top_stocks))
        quantity = int(allocation / entry)

        portfolio.append({
            "Stock": row['Stock'],
            "Entry": round(entry,2),
            "SL": round(entry - atr*2,2),
            "Target": round(entry + atr*4,2),
            "Qty": quantity
        })

# ==============================
# 🔹 STEP 7: MESSAGE
# ==============================
message = "📊 AI STOCK SIGNALS\n\n"

message += f"🧠 Mode: {market_phase}\n\n"
message += "📈 MARKET: UPTREND ✅\n\n" if market_uptrend else "📉 MARKET: DOWNTREND ❌\n\n"

# Watchlist
if len(results) > 0:
    message += "👀 WATCHLIST:\n"
    for _, row in df.head(5).iterrows():
        message += f"{row['Stock']} - {round(row['Probability'],2)}\n"
    message += "\n"

# Portfolio
if portfolio:
    message += "💼 PORTFOLIO:\n\n"

    for p in portfolio:
        
        message += (
            f"{p['Stock']}\n"
            f"Entry: ₹{p['Entry']}\n"
            f"SL: ₹{p['SL']}\n"
            f"Target: ₹{p['Target']}\n"
            f"Qty: {p['Qty']}\n"
            f"R:R = 1:2\n\n"
    )
else:
    message += "No trades today\n"

# ==============================
# 🔹 STEP 8: TELEGRAM
# ==============================
response = requests.post(
    f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
    data={"chat_id": CHAT_ID, "text": message}
)

print("Telegram:", response.text)