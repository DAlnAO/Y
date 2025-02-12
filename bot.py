import requests
import pandas as pd
import numpy as np
import time
import schedule
import telebot
import ta
from datetime import datetime

# Telegram 机器人 Token 和 Chat ID（请替换为你的）
TELEGRAM_BOT_TOKEN = "your_telegram_bot_token"
TELEGRAM_CHAT_ID = "your_chat_id"
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# OKX API 获取 K 线数据
def get_okx_data(symbol, timeframe="15m", limit=200):
    url = f"https://www.okx.com/api/v5/market/candles?instId={symbol}&bar={timeframe}&limit={limit}"
    response = requests.get(url)
    data = response.json()
    
    if "data" in data:
        df = pd.DataFrame(data["data"], columns=["timestamp", "open", "high", "low", "close", "volume", "_"])
        df = df.drop(columns=["_"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
        return df[::-1].reset_index(drop=True)
    return None

# 计算技术指标
def calculate_indicators(df):
    df["SMA_50"] = ta.trend.sma_indicator(df["close"], window=50)
    df["SMA_200"] = ta.trend.sma_indicator(df["close"], window=200)
    
    macd = ta.trend.MACD(df["close"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    
    df["RSI"] = ta.momentum.RSIIndicator(df["close"]).rsi()
    
    adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"])
    df["ADX"] = adx.adx()
    
    df["ATR"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range()
    
    bollinger = ta.volatility.BollingerBands(df["close"])
    df["BB_upper"] = bollinger.bollinger_hband()
    df["BB_lower"] = bollinger.bollinger_lband()
    
    df["VWAP"] = ta.volume.VolumeWeightedAveragePrice(df["high"], df["low"], df["close"], df["volume"]).volume_weighted_average_price()
    
    return df

# 筛选符合交易策略的币种
def filter_trading_opportunities(symbol):
    df = get_okx_data(symbol)
    if df is None:
        return None
    
    df = calculate_indicators(df)
    
    latest = df.iloc[-1]
    close_price = latest["close"]
    atr = latest["ATR"]
    
    # 做多信号
    long_conditions = [
        latest["SMA_50"] > latest["SMA_200"],
        latest["MACD"] > latest["MACD_signal"],
        latest["RSI"] > 50,
        latest["ADX"] > 25,
        latest["close"] > latest["BB_lower"],
        latest["close"] > latest["VWAP"]
    ]
    
    # 做空信号
    short_conditions = [
        latest["SMA_50"] < latest["SMA_200"],
        latest["MACD"] < latest["MACD_signal"],
        latest["RSI"] < 50,
        latest["ADX"] > 25,
        latest["close"] < latest["BB_upper"],
        latest["close"] < latest["VWAP"]
    ]

    if all(long_conditions):
        return {
            "symbol": symbol,
            "side": "做多",
            "entry": close_price,
            "stop_loss": close_price - 2 * atr,
            "take_profit": close_price + 4 * atr
        }
    elif all(short_conditions):
        return {
            "symbol": symbol,
            "side": "做空",
            "entry": close_price,
            "stop_loss": close_price + 2 * atr,
            "take_profit": close_price - 4 * atr
        }
    
    return None

# 获取 OKX 可交易合约列表
def get_okx_contracts():
    url = "https://www.okx.com/api/v5/public/instruments?instType=SWAP"
    response = requests.get(url)
    data = response.json()
    
    if "data" in data:
        return [item["instId"] for item in data["data"]]
    return []

# 运行策略，选择最佳 3 个交易标的
def run_strategy():
    contracts = get_okx_contracts()
    potential_trades = []

    for symbol in contracts[:20]:  
        trade_info = filter_trading_opportunities(symbol)
        if trade_info:
            potential_trades.append(trade_info)

    if potential_trades:
        message = "📊 **OKX 合约交易策略** 📊\n"
        for trade in potential_trades[:3]:
            message += f"🔹 交易对: {trade['symbol']}\n"
            message += f"📈 方向: {trade['side']}\n"
            message += f"🎯 进场价格: {trade['entry']:.2f}\n"
            message += f"⛔ 止损: {trade['stop_loss']:.2f}\n"
            message += f"🎯 止盈: {trade['take_profit']:.2f}\n\n"

        bot.send_message(TELEGRAM_CHAT_ID, message)
    else:
        bot.send_message(TELEGRAM_CHAT_ID, "当前市场无符合策略的合约交易机会")

# 每 30 分钟运行一次
schedule.every(30).minutes.do(run_strategy)

if __name__ == "__main__":
    print("OKX 合约交易策略机器人启动...")
    while True:
        schedule.run_pending()
        time.sleep(1)