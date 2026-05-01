import json
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import ta
import yfinance as yf
from dotenv import load_dotenv
from sklearn.utils import resample
from xgboost import XGBClassifier


# ==============================
# STEP 0: CONFIG
# ==============================
load_dotenv()

CAPITAL = float(os.getenv("CAPITAL", 100000))
MAX_PER_STOCK = float(os.getenv("MAX_PER_STOCK", 0.4))
MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", 5))
PAPER_TRADING = os.getenv("PAPER_TRADING", "true").lower() in {"1", "true", "yes", "on"}
SLIPPAGE = float(os.getenv("SLIPPAGE", 0.001))
COOLDOWN_DAYS = int(os.getenv("COOLDOWN_DAYS", 1))
LIVE_PRICE_TTL_SECONDS = int(os.getenv("LIVE_PRICE_TTL_SECONDS", 60))
YF_RETRIES = int(os.getenv("YF_RETRIES", 2))
YF_RETRY_SLEEP_SECONDS = float(os.getenv("YF_RETRY_SLEEP_SECONDS", 1))
MAX_EQUITY_POINTS = int(os.getenv("MAX_EQUITY_POINTS", 500))

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

ALERTS_FILE = "alerts.json"
PAPER_FILE = "paper_trades.json"
REPORT_FILE = "performance_report.json"
LIVE_PRICE_CACHE = {}
REQUIRED_OHLCV_COLUMNS = {"Open", "High", "Low", "Close", "Volume"}


# ==============================
# HELPERS
# ==============================
def load_json_file(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save_json_file(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def as_float(value, default=0.0):
    try:
        if isinstance(value, pd.Series):
            value = value.iloc[-1]
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def apply_exit_slippage(price):
    return round(as_float(price) * (1 - SLIPPAGE), 2)


def parse_date(value):
    try:
        return datetime.fromisoformat(value).date()
    except (TypeError, ValueError):
        return None


def max_drawdown_from_equity(equity_values):
    if not equity_values:
        return 0

    peak = equity_values[0]
    max_dd = 0
    for equity in equity_values:
        peak = max(peak, equity)
        if peak > 0:
            max_dd = min(max_dd, (equity - peak) / peak)
    return max_dd


def has_required_columns(data, columns=REQUIRED_OHLCV_COLUMNS):
    return not data.empty and columns.issubset(set(data.columns))


def normalize_columns(data):
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


def safe_yf_download(symbol, **kwargs):
    last_error = None

    for attempt in range(YF_RETRIES + 1):
        try:
            data = yf.download(symbol, **kwargs)
            data = normalize_columns(data)
            if not data.empty:
                return data
        except Exception as exc:
            last_error = exc

        if attempt < YF_RETRIES:
            time.sleep(YF_RETRY_SLEEP_SECONDS)

    if last_error:
        print(f"Download failed for {symbol}: {last_error}")
    return pd.DataFrame()


def get_market_phase():
    now = datetime.now(timezone.utc)
    india_hour = now.hour + 5.5

    if 9 <= india_hour < 15.5:
        return "LIVE"
    if 8 <= india_hour < 9:
        return "PRE"
    return "CLOSED"


def get_nse_stocks():
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    df = pd.read_csv(url)
    return [symbol + ".NS" for symbol in df["SYMBOL"].tolist()]


def filter_liquid_stocks(stocks):
    filtered = []

    for stock in stocks[:300]:
        try:
            data = safe_yf_download(stock, period="5d", progress=False)

            if not has_required_columns(data, {"Close", "Volume"}):
                continue

            price = as_float(data["Close"].iloc[-1])
            volume = as_float(data["Volume"].rolling(5).mean().iloc[-1])

            if 50 < price < 2000 and volume > 500000:
                filtered.append(stock)
        except Exception as exc:
            print(f"Liquidity check failed for {stock}: {exc}")

    return filtered


def get_live_price(symbol):
    cached = LIVE_PRICE_CACHE.get(symbol)
    now = time.time()
    if cached and now - cached["timestamp"] <= LIVE_PRICE_TTL_SECONDS:
        return cached["price"]

    try:
        data = safe_yf_download(symbol, period="1d", interval="1m", progress=False)
        if not data.empty:
            price = as_float(data["Close"].iloc[-1], None)
            if price is not None and price > 0:
                LIVE_PRICE_CACHE[symbol] = {"price": price, "timestamp": now}
                return price
    except Exception:
        pass
    return None


def send_telegram_message(message):
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram skipped: BOT_TOKEN or CHAT_ID missing")
        return None

    response = requests.post(
        f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
        data={"chat_id": CHAT_ID, "text": message},
        timeout=20,
    )
    print("Telegram:", response.text)
    return response


# ==============================
# INDICATORS + STRATEGY
# ==============================
def add_indicators(data, nifty_close):
    data = data.copy()
    data = data.replace([float("inf"), float("-inf")], pd.NA)
    close = data["Close"]

    data["RSI"] = ta.momentum.RSIIndicator(close).rsi()
    data["MA20"] = close.rolling(20).mean()
    data["MA50"] = close.rolling(50).mean()
    data["MA200"] = close.rolling(200).mean()
    data["Vol_Avg"] = data["Volume"].rolling(20).mean()
    data["Returns"] = close.pct_change()
    data["Volatility"] = data["Returns"].rolling(10).std()
    data["Trend"] = data["MA20"] - data["MA50"]

    macd = ta.trend.MACD(close)
    data["MACD"] = macd.macd()
    data["MACD_signal"] = macd.macd_signal()
    data["MACD_hist"] = macd.macd_diff()

    atr = ta.volatility.AverageTrueRange(data["High"], data["Low"], close)
    data["ATR"] = atr.average_true_range()

    data["Nifty_Return_20"] = nifty_close.pct_change(20).reindex(data.index).ffill()
    data["Stock_Return_20"] = close.pct_change(20)
    data["Relative_Strength"] = data["Stock_Return_20"] - data["Nifty_Return_20"]

    data["Entry"] = (
        (close > data["MA20"])
        & (data["MA20"] > data["MA50"])
        & (data["MA50"] > data["MA200"])
        & data["RSI"].between(52, 70)
        & (data["MACD"] > data["MACD_signal"])
        & (data["MACD_hist"] > data["MACD_hist"].shift(1))
        & (data["Volume"] > data["Vol_Avg"] * 1.2)
        & (data["Relative_Strength"] > 0)
    )
    data["Exit"] = (
        (close < data["MA20"])
        | (data["RSI"] < 45)
        | (data["MACD"] < data["MACD_signal"])
        | (data["Relative_Strength"] < -0.02)
    )

    data["Future_Return"] = close.shift(-5) / close - 1
    data["Target"] = (
        (data["Future_Return"] > 0.025)
        & (data["RSI"] > 50)
        & (data["Relative_Strength"] > 0)
    ).astype(int)

    return data


def backtest_strategy(data):
    position = None
    equity = 1.0
    equity_curve = []
    trades = []

    for date, row in data.iterrows():
        close = as_float(row["Close"])
        atr = as_float(row["ATR"])

        if position is None:
            if bool(row["Entry"]) and atr > 0:
                position = {
                    "entry_date": date,
                    "entry": close,
                    "sl": close - atr * 2,
                    "target": close + atr * 4,
                    "highest": close,
                }
            equity_curve.append(equity)
            continue

        position["highest"] = max(position["highest"], close)
        trailing_sl = position["highest"] - atr * 2.5
        effective_sl = max(position["sl"], trailing_sl)

        exit_reason = None
        exit_price = close
        if close <= effective_sl:
            exit_reason = "SL"
            exit_price = apply_exit_slippage(effective_sl)
        elif close >= position["target"]:
            exit_reason = "TARGET"
            exit_price = apply_exit_slippage(position["target"])
        elif bool(row["Exit"]):
            exit_reason = "SIGNAL"
            exit_price = apply_exit_slippage(close)

        if exit_reason:
            trade_return = exit_price / position["entry"] - 1
            equity *= 1 + trade_return
            trades.append(
                {
                    "entry_date": str(position["entry_date"].date()),
                    "exit_date": str(date.date()),
                    "entry": round(position["entry"], 2),
                    "exit": round(exit_price, 2),
                    "return": trade_return,
                    "reason": exit_reason,
                }
            )
            position = None

        equity_curve.append(equity)

    bt = data.copy()
    bt["Cumulative_Strategy"] = equity_curve
    bt["Strategy_Return"] = bt["Cumulative_Strategy"].pct_change().fillna(0)
    return bt, trades


def calculate_metrics(bt, trades):
    if bt.empty:
        return {
            "total_return": 0,
            "win_rate": 0,
            "max_dd": 0,
            "sharpe": 0,
            "trades": 0,
            "profit_factor": 0,
            "avg_trade_return": 0,
        }

    returns = bt["Strategy_Return"].dropna()
    total_return = as_float(bt["Cumulative_Strategy"].iloc[-1] - 1)
    peak = bt["Cumulative_Strategy"].cummax()
    drawdown = (bt["Cumulative_Strategy"] - peak) / peak
    max_dd = as_float(drawdown.min())
    sharpe = as_float((returns.mean() / returns.std()) * (252**0.5)) if returns.std() != 0 else 0

    trade_returns = [trade["return"] for trade in trades]
    wins = [ret for ret in trade_returns if ret > 0]
    losses = [ret for ret in trade_returns if ret < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))

    return {
        "total_return": total_return,
        "win_rate": len(wins) / len(trade_returns) if trade_returns else 0,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "trades": len(trade_returns),
        "profit_factor": gross_profit / gross_loss if gross_loss else 0,
        "avg_trade_return": sum(trade_returns) / len(trade_returns) if trade_returns else 0,
    }


def check_intraday_crash(stock):
    try:
        data = safe_yf_download(stock, period="1d", interval="5m", progress=False)

        if not has_required_columns(data, {"Close", "Volume"}) or len(data) < 20:
            return None

        close = data["Close"]
        recent = close.iloc[-3:]
        change = (recent.iloc[-1] - recent.iloc[0]) / recent.iloc[0] * 100

        vol = data["Volume"]
        recent_vol = as_float(vol.iloc[-1])
        avg_vol = as_float(vol.rolling(20).mean().iloc[-1])

        if change <= -1.5 and recent_vol > avg_vol:
            return round(as_float(change), 2)
    except Exception as exc:
        print(f"Error in crash check for {stock}: {exc}")

    return None


# ==============================
# PAPER TRADING + PNL
# ==============================
def default_paper_state():
    return {
        "starting_capital": CAPITAL,
        "cash": CAPITAL,
        "open_positions": [],
        "closed_trades": [],
        "equity_curve": [{"date": datetime.now().date().isoformat(), "equity": CAPITAL}],
    }


def load_paper_state():
    state = load_json_file(PAPER_FILE, default_paper_state())
    if not isinstance(state, dict):
        state = default_paper_state()

    state.setdefault("starting_capital", CAPITAL)
    state.setdefault("cash", CAPITAL)
    state.setdefault("open_positions", [])
    state.setdefault("closed_trades", [])
    state.setdefault("equity_curve", [{"date": datetime.now().date().isoformat(), "equity": state["cash"]}])
    state["equity_curve"] = state["equity_curve"][-MAX_EQUITY_POINTS:]
    return state


def update_paper_positions(state):
    remaining = []
    today = datetime.now().date().isoformat()

    for position in state["open_positions"]:
        current_price = get_live_price(position["stock"])
        if current_price is None:
            position["last_price"] = position.get("last_price", position["entry"])
            remaining.append(position)
            continue

        position["last_price"] = round(current_price, 2)
        position["highest"] = max(position.get("highest", position["entry"]), current_price)

        trailing_sl = position["highest"] - position["atr"] * 2.5
        position["sl"] = round(max(position["sl"], trailing_sl), 2)

        exit_reason = None
        exit_price = current_price

        if current_price <= position["sl"]:
            exit_reason = "SL"
            exit_price = apply_exit_slippage(position["sl"])
        elif current_price >= position["target"]:
            exit_reason = "TARGET"
            exit_price = apply_exit_slippage(position["target"])

        if exit_reason:
            pnl = (exit_price - position["entry"]) * position["qty"]
            state["cash"] += exit_price * position["qty"]
            state["closed_trades"].append(
                {
                    **position,
                    "exit": round(exit_price, 2),
                    "exit_date": today,
                    "exit_reason": exit_reason,
                    "pnl": round(pnl, 2),
                    "pnl_pct": round((exit_price / position["entry"] - 1) * 100, 2),
                }
            )
        else:
            remaining.append(position)

    state["open_positions"] = remaining
    return state


def is_in_cooldown(state, stock, today):
    for trade in reversed(state["closed_trades"]):
        if trade.get("stock") != stock:
            continue

        exit_date = parse_date(trade.get("exit_date"))
        if exit_date and today - exit_date < timedelta(days=COOLDOWN_DAYS):
            return True

    return False


def open_paper_positions(state, portfolio, market_phase):
    if market_phase != "LIVE":
        print("Market is not LIVE. New paper trades blocked.")
        return state

    open_symbols = {position["stock"] for position in state["open_positions"]}
    today_date = datetime.now().date()
    today = today_date.isoformat()

    for trade in portfolio:
        if len(state["open_positions"]) >= MAX_OPEN_POSITIONS:
            break

        if trade["Stock"] in open_symbols or is_in_cooldown(state, trade["Stock"], today_date):
            continue

        entry_price = get_live_price(trade["Stock"])
        if entry_price is None:
            entry_price = trade["Entry"]
        entry_price = round(as_float(entry_price), 2)

        if entry_price <= 0:
            continue

        allocation = min(CAPITAL * MAX_PER_STOCK, state["cash"])
        quantity = int(allocation / entry_price)
        if quantity <= 0:
            continue

        cost = entry_price * quantity
        if cost > state["cash"]:
            continue

        atr = as_float(trade["ATR"])
        if atr <= 0:
            continue

        state["cash"] -= cost
        state["open_positions"].append(
            {
                "stock": trade["Stock"],
                "entry": entry_price,
                "sl": round(entry_price - atr * 2, 2),
                "target": round(entry_price + atr * 4, 2),
                "qty": quantity,
                "atr": atr,
                "highest": entry_price,
                "last_price": entry_price,
                "entry_date": today,
            }
        )
        open_symbols.add(trade["Stock"])

    return state


def build_performance_report(state, signal_metrics):
    realized_pnl = sum(trade.get("pnl", 0) for trade in state["closed_trades"])
    unrealized_pnl = sum(
        (position.get("last_price", position["entry"]) - position["entry"]) * position["qty"]
        for position in state["open_positions"]
    )
    equity = state["cash"] + sum(
        position.get("last_price", position["entry"]) * position["qty"]
        for position in state["open_positions"]
    )
    closed = state["closed_trades"]
    wins = [trade for trade in closed if trade.get("pnl", 0) > 0]
    losses = [trade for trade in closed if trade.get("pnl", 0) < 0]
    gross_profit = sum(trade["pnl"] for trade in wins)
    gross_loss = abs(sum(trade["pnl"] for trade in losses))
    avg_trade_return = (
        sum(trade.get("pnl_pct", 0) for trade in closed) / len(closed) if closed else 0
    )

    state["equity_curve"].append(
        {
            "date": datetime.now().isoformat(timespec="seconds"),
            "equity": round(equity, 2),
        }
    )
    state["equity_curve"] = state["equity_curve"][-MAX_EQUITY_POINTS:]
    equity_values = [as_float(point.get("equity")) for point in state["equity_curve"]]

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "paper_trading": PAPER_TRADING,
        "starting_capital": round(state["starting_capital"], 2),
        "cash": round(state["cash"], 2),
        "equity": round(equity, 2),
        "realized_pnl": round(realized_pnl, 2),
        "unrealized_pnl": round(unrealized_pnl, 2),
        "total_pnl": round(realized_pnl + unrealized_pnl, 2),
        "return_pct": round((equity / state["starting_capital"] - 1) * 100, 2),
        "open_positions": len(state["open_positions"]),
        "total_trades": len(closed),
        "closed_trades": len(closed),
        "win_rate": round(len(wins) / len(closed) * 100, 2) if closed else 0,
        "paper_win_rate": round(len(wins) / len(closed) * 100, 2) if closed else 0,
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss else 0,
        "paper_profit_factor": round(gross_profit / gross_loss, 2) if gross_loss else 0,
        "avg_trade_return": round(avg_trade_return, 2),
        "max_drawdown": round(max_drawdown_from_equity(equity_values) * 100, 2),
        "equity_curve": state["equity_curve"],
        "latest_signal_metrics": signal_metrics,
    }
    save_json_file(REPORT_FILE, report)
    return report


def format_money(value):
    return f"Rs {round(value, 2)}"


# ==============================
# MAIN
# ==============================
alerted_stocks_data = load_json_file(ALERTS_FILE, [])
alerted_stocks = set(alerted_stocks_data if isinstance(alerted_stocks_data, list) else [])

market_phase = get_market_phase()
print("Market Phase:", market_phase)
print("Paper Trading:", "ON" if PAPER_TRADING else "OFF")

stocks = get_nse_stocks()
stocks = filter_liquid_stocks(stocks)
stocks = stocks[:200]
print("Total stocks loaded:", len(stocks))

nifty = safe_yf_download("^NSEI", period="1y", progress=False)
if nifty.empty or "Close" not in nifty:
    raise RuntimeError("Unable to load Nifty data")

nifty_close = nifty["Close"].squeeze()
nifty_ma50 = nifty_close.rolling(50).mean()

last_close = as_float(nifty_close.iloc[-1])
last_ma50 = as_float(nifty_ma50.iloc[-1], None)
market_uptrend = False if last_ma50 is None else last_close > last_ma50

results = []

for stock in stocks:
    crash = check_intraday_crash(stock)

    if crash and stock not in alerted_stocks:
        print(f"Crash detected in {stock}: {crash}%")
        alerted_stocks.add(stock)
        save_json_file(ALERTS_FILE, sorted(alerted_stocks))

        results.append(
            {
                "Stock": stock,
                "Probability": 1.0,
                "ATR": 0,
                "Price": 0,
                "Strategy_Return": 0,
                "WinRate": 0,
                "Sharpe": 0,
                "MaxDD": 0,
                "Trades": 0,
                "ProfitFactor": 0,
                "AvgTradeReturn": 0,
                "Type": "CRASH",
                "Crash": crash,
            }
        )
        continue

    print(f"Processing {stock}...")

    try:
        data = safe_yf_download(stock, period="2y", progress=False)

        if not has_required_columns(data) or len(data) < 200:
            continue

        data = add_indicators(data, nifty_close)
        last_close_price = as_float(data["Close"].iloc[-1])
        live_price = get_live_price(stock)
        price = as_float(live_price, last_close_price) if live_price is not None else last_close_price
        avg_volume = as_float(data["Volume"].rolling(20).mean().iloc[-1])

        if price < 50 or price > 2000 or avg_volume < 500000:
            continue

        latest = data.iloc[-1]
        trend_strength = as_float((latest["MA20"] - latest["MA50"]) / latest["MA50"])
        if trend_strength < 0.005:
            continue

        if not bool(latest["Entry"]):
            continue

        features = [
            "RSI",
            "MACD",
            "MACD_signal",
            "MACD_hist",
            "MA20",
            "MA50",
            "MA200",
            "ATR",
            "Volume",
            "Volatility",
            "Trend",
            "Relative_Strength",
        ]
        data = data.replace([float("inf"), float("-inf")], pd.NA)
        data = data.dropna(subset=features + ["Target", "Entry", "Exit"])
        if len(data) < 200:
            continue

        bt, trades = backtest_strategy(data)
        metrics = calculate_metrics(bt, trades)

        df_major = data[data["Target"] == 0]
        df_minor = data[data["Target"] == 1]

        if len(df_minor) == 0 or len(df_major) == 0:
            continue

        df_minor = resample(df_minor, replace=True, n_samples=len(df_major), random_state=42)
        balanced = pd.concat([df_major, df_minor]).sample(frac=1, random_state=42)

        X = balanced[features]
        y = balanced["Target"]
        split = int(len(X) * 0.8)

        if split <= 0 or split >= len(X):
            continue

        model = XGBClassifier(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
        )
        model.fit(X.iloc[:split], y.iloc[:split])

        prob = as_float(model.predict_proba(data[features].iloc[-1:])[:, 1][0])
        if not market_uptrend:
            prob *= 0.75

        if prob < 0.6 or metrics["sharpe"] < 0.5 or metrics["win_rate"] < 0.5:
            continue

        atr = as_float(latest["ATR"])
        risk = atr * 2
        reward = atr * 4

        if atr <= 0 or reward / risk < 2:
            continue

        print(f"Adding {stock} to results")
        results.append(
            {
                "Stock": stock,
                "Probability": prob,
                "ATR": atr,
                "Price": price,
                "Strategy_Return": metrics["total_return"],
                "WinRate": metrics["win_rate"],
                "Sharpe": metrics["sharpe"],
                "MaxDD": metrics["max_dd"],
                "Trades": metrics["trades"],
                "ProfitFactor": metrics["profit_factor"],
                "AvgTradeReturn": metrics["avg_trade_return"],
            }
        )
    except Exception as exc:
        print(f"Error in {stock}: {exc}")

print("Total selected stocks:", len(results))

if len(results) == 0:
    print("No ML signals, using fallback")
    for stock in stocks[:5]:
        results.append(
            {
                "Stock": stock,
                "Probability": 0.5,
                "ATR": 10,
                "Price": 100,
                "Strategy_Return": 0.01,
                "WinRate": 0.5,
                "Sharpe": 0.5,
                "MaxDD": 0,
                "Trades": 0,
                "ProfitFactor": 0,
                "AvgTradeReturn": 0,
                "Type": "FALLBACK",
            }
        )

df = pd.DataFrame(results)
portfolio = []

if not df.empty:
    if "Type" in df.columns:
        non_crash = df[~df["Type"].isin(["CRASH", "FALLBACK"])].copy()
    else:
        non_crash = df.copy()

    if not non_crash.empty:
        non_crash["Score"] = (
            non_crash["Probability"] * 0.35
            + non_crash["Sharpe"].clip(lower=-2, upper=4) * 0.2
            + non_crash["Strategy_Return"].clip(lower=-1, upper=2) * 0.2
            + non_crash["WinRate"] * 0.15
            + non_crash["ProfitFactor"].clip(upper=5) * 0.1
        )
        non_crash = non_crash.sort_values(by="Score", ascending=False)

        top_stocks = non_crash.head(MAX_OPEN_POSITIONS)

        for _, row in top_stocks.iterrows():
            live_entry = get_live_price(row["Stock"])
            entry = as_float(live_entry, as_float(row["Price"])) if live_entry is not None else as_float(row["Price"])
            atr = as_float(row["ATR"])
            allocation = min(CAPITAL * MAX_PER_STOCK, CAPITAL / max(len(top_stocks), 1))
            quantity = int(allocation / entry) if entry > 0 else 0

            portfolio.append(
                {
                    "Stock": row["Stock"],
                    "Entry": round(entry, 2),
                    "SL": round(entry - atr * 2, 2),
                    "Target": round(entry + atr * 4, 2),
                    "Qty": quantity,
                    "ATR": round(atr, 2),
                    "Probability": round(as_float(row["Probability"]), 2),
                    "WinRate": round(as_float(row["WinRate"]) * 100, 2),
                    "Sharpe": round(as_float(row["Sharpe"]), 2),
                    "Strategy_Return": round(as_float(row["Strategy_Return"]) * 100, 2),
                }
            )

signal_metrics = {
    "selected": len(portfolio),
    "avg_probability": round(as_float(pd.Series([p["Probability"] for p in portfolio]).mean()), 2)
    if portfolio
    else 0,
    "avg_win_rate": round(as_float(pd.Series([p["WinRate"] for p in portfolio]).mean()), 2)
    if portfolio
    else 0,
}

paper_state = load_paper_state()
if PAPER_TRADING:
    paper_state = update_paper_positions(paper_state)
    paper_state = open_paper_positions(paper_state, portfolio, market_phase)
    save_json_file(PAPER_FILE, paper_state)

report = build_performance_report(paper_state, signal_metrics)
if PAPER_TRADING:
    save_json_file(PAPER_FILE, paper_state)

# ==============================
# MESSAGE
# ==============================
message = "AI STOCK SIGNALS\n\n"
message += f"Mode: {market_phase}\n"
message += f"Paper Trading: {'ON' if PAPER_TRADING else 'OFF'}\n"
message += "MARKET: UPTREND\n\n" if market_uptrend else "MARKET: DOWNTREND\n\n"

message += "WATCHLIST:\n"
for _, row in df.head(5).iterrows():
    if row.get("Type") == "CRASH":
        message += f"{row['Stock']} - CRASH {row['Crash']}%\n"
    elif row.get("Type") == "FALLBACK":
        message += f"{row['Stock']} - fallback watch only\n"
    else:
        message += (
            f"{row['Stock']} - Prob {round(as_float(row['Probability']), 2)} | "
            f"Win {round(as_float(row['WinRate']) * 100, 1)}% | "
            f"Sharpe {round(as_float(row['Sharpe']), 2)}\n"
        )

if portfolio:
    message += "\nPORTFOLIO:\n\n"
    for trade in portfolio:
        message += (
            f"{trade['Stock']}\n"
            f"Entry: {format_money(trade['Entry'])}\n"
            f"SL: {format_money(trade['SL'])}\n"
            f"Target: {format_money(trade['Target'])}\n"
            f"Qty: {trade['Qty']}\n"
            f"R:R = 1:2\n\n"
        )
else:
    message += "\nNo trades today\n"

message += "PERFORMANCE:\n"
message += f"Equity: {format_money(report['equity'])}\n"
message += f"Realized PnL: {format_money(report['realized_pnl'])}\n"
message += f"Unrealized PnL: {format_money(report['unrealized_pnl'])}\n"
message += f"Total Return: {report['return_pct']}%\n"
message += f"Closed Trades: {report['closed_trades']} | Win Rate: {report['paper_win_rate']}%\n"

send_telegram_message(message)
print(message)
