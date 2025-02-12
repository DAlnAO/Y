import logging
import requests
import pandas as pd
import ta
from datetime import datetime
import time
import schedule

# 设置日志记录到文件
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(message)s', 
                    handlers=[logging.FileHandler('trading_bot.log', 'a', 'utf-8')])
logger = logging.getLogger()

# OKX API 获取 K 线数据
def get_okx_data(symbol, timeframe="15m", limit=200):
    try:
        url = f"https://www.okx.com/api/v5/market/candles?instId={symbol}&bar={timeframe}&limit={limit}"
        response = requests.get(url)
        data = response.json()
        
        if "data" in data:
            df = pd.DataFrame(data["data"], columns=["timestamp", "open", "high", "low", "close", "volume", "_"])
            df = df.drop(columns=["_"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
            df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
            return df[::-1].reset_index(drop=True)
        else:
            logger.warning(f"获取数据失败: {symbol} 没有返回数据")
            return None
    except Exception as e:
        logger.error(f"获取 OKX 数据时发生错误: {e}")
        return None

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
    try:
        url = "https://www.okx.com/api/v5/public/instruments?instType=SWAP"
        response = requests.get(url)
        data = response.json()
        
        if "data" in data:
            return [item["instId"] for item in data["data"]]
        else:
            logger.warning("获取 OKX 合约列表失败")
            return []
    except Exception as e:
        logger.error(f"获取 OKX 合约列表时发生错误: {e}")
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

        # 将策略输出记录到日志文件
        logger.info(message)
    else:
        logger.info("当前市场无符合策略的合约交易机会")

# 每 30 分钟运行一次
schedule.every(30).minutes.do(run_strategy)

if __name__ == "__main__":
    logger.info("OKX 合约交易策略机器人启动...")
    while True:
        schedule.run_pending()
        time.sleep(1)