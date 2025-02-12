import logging
import requests
import pandas as pd
import ta
from datetime import datetime
import time
import schedule
import threading

# 设置日志记录到文件
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(message)s', 
                    handlers=[logging.FileHandler('trading_bot.log', 'a', 'utf-8')])
logger = logging.getLogger()

# OKX API 获取 K 线数据
def get_okx_data(symbol, timeframe="15m", limit=200):
    try:
        url = f"https://www.okx.com/api/v5/market/candles?instId={symbol}&bar={timeframe}&limit={limit}"
        logger.info(f"【获取数据】 请求 URL: {url}")
        response = requests.get(url)
        data = response.json()
        
        if "data" in data:
            df = pd.DataFrame(data["data"], columns=["timestamp", "open", "high", "low", "close", "volume", "close_ask", "close_bid", "instrument_id"])
            df = df.drop(columns=["instrument_id", "close_ask", "close_bid"])  # 移除不必要的列
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
            df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
            logger.info(f"【获取数据】 成功获取 {symbol} 数据，共 {len(df)} 行")
            return df[::-1].reset_index(drop=True)
        else:
            logger.warning(f"【获取数据失败】 {symbol} 没有返回数据")
            return None
    except Exception as e:
        logger.error(f"【获取数据失败】 获取 OKX 数据时发生错误: {e}")
        return None

# 计算指标
def calculate_indicators(df):
    logger.info("【计算指标】开始计算技术指标")
    df["SMA_50"] = ta.trend.sma_indicator(df["close"], window=50)
    df["SMA_200"] = ta.trend.sma_indicator(df["close"], window=200)
    df["MACD"] = ta.trend.macd(df["close"])
    df["MACD_signal"] = ta.trend.macd_signal(df["close"])
    df["RSI"] = ta.momentum.rsi(df["close"])
    df["ADX"] = ta.trend.adx(df["high"], df["low"], df["close"])
    df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"])
    df["BB_upper"], df["BB_middle"], df["BB_lower"] = ta.volatility.bollinger_hband(df["close"]), ta.volatility.bollinger_mavg(df["close"]), ta.volatility.bollinger_lband(df["close"])
    df["VWAP"] = ta.volume.volume_weighted_average_price(df["high"], df["low"], df["close"], df["volume"], window=14)

    logger.info(f"【计算指标】完成技术指标计算")
    return df

# 筛选符合交易策略的币种（结合 5 分钟和 15 分钟 K 线）
def filter_trading_opportunities(symbol):
    logger.info(f"【筛选交易机会】开始分析币种: {symbol}")
    
    # 获取 5m 和 15m K 线数据
    df_5m = get_okx_data(symbol, timeframe="5m")
    df_15m = get_okx_data(symbol, timeframe="15m")
    
    if df_5m is None or df_15m is None:
        logger.info(f"【筛选交易机会】{symbol} 数据为空，跳过")
        return None
    
    # 计算指标
    df_5m = calculate_indicators(df_5m)
    df_15m = calculate_indicators(df_15m)
    
    latest_5m = df_5m.iloc[-1]
    latest_15m = df_15m.iloc[-1]
    
    close_price = latest_15m["close"]
    atr = latest_15m["ATR"]

    # 组合 5m 和 15m 信号
    long_conditions = [
        latest_5m["SMA_50"] > latest_5m["SMA_200"] and latest_15m["SMA_50"] > latest_15m["SMA_200"],
        latest_5m["MACD"] > latest_5m["MACD_signal"] and latest_15m["MACD"] > latest_15m["MACD_signal"],
        latest_5m["RSI"] > 50 and latest_15m["RSI"] > 50,
        latest_5m["ADX"] > 25 and latest_15m["ADX"] > 25,
        latest_5m["close"] > latest_5m["BB_lower"] and latest_15m["close"] > latest_15m["BB_lower"],
        latest_5m["close"] > latest_5m["VWAP"] and latest_15m["close"] > latest_15m["VWAP"]
    ]
    
    short_conditions = [
        latest_5m["SMA_50"] < latest_5m["SMA_200"] and latest_15m["SMA_50"] < latest_15m["SMA_200"],
        latest_5m["MACD"] < latest_5m["MACD_signal"] and latest_15m["MACD"] < latest_15m["MACD_signal"],
        latest_5m["RSI"] < 50 and latest_15m["RSI"] < 50,
        latest_5m["ADX"] > 25 and latest_15m["ADX"] > 25,
        latest_5m["close"] < latest_5m["BB_upper"] and latest_15m["close"] < latest_15m["BB_upper"],
        latest_5m["close"] < latest_5m["VWAP"] and latest_15m["close"] < latest_15m["VWAP"]
    ]

    # 记录每个条件的计算结果
    logger.info(f"【筛选交易机会】{symbol} 做多条件: {long_conditions}")
    logger.info(f"【筛选交易机会】{symbol} 做空条件: {short_conditions}")

    if all(long_conditions):
        logger.info(f"【筛选交易机会】{symbol} 符合做多条件")
        return {
            "symbol": symbol,
            "side": "做多",
            "entry": close_price,
            "stop_loss": close_price - 2 * atr,
            "take_profit": close_price + 4 * atr
        }
    elif all(short_conditions):
        logger.info(f"【筛选交易机会】{symbol} 符合做空条件")
        return {
            "symbol": symbol,
            "side": "做空",
            "entry": close_price,
            "stop_loss": close_price + 2 * atr,
            "take_profit": close_price - 4 * atr
        }

    logger.info(f"【筛选交易机会】{symbol} 不符合交易条件")
    return None

# 获取 OKX 可交易合约列表
def get_okx_contracts():
    try:
        url = "https://www.okx.com/api/v5/public/instruments?instType=SWAP"
        logger.info("【获取合约列表】 请求获取 OKX 可交易合约列表")
        response = requests.get(url)
        data = response.json()
        
        if "data" in data:
            logger.info(f"【获取合约列表】成功获取 OKX 合约列表，共 {len(data['data'])} 个合约")
            return [item["instId"] for item in data["data"]]
        else:
            logger.warning("【获取合约列表】获取 OKX 合约列表失败")
            return []
    except Exception as e:
        logger.error(f"【获取合约列表】 获取 OKX 合约列表时发生错误: {e}")
        return []

# 运行策略，选择最佳 3 个交易标的
def run_strategy():
    logger.info("【运行策略】开始运行策略...")
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
        logger.info(f"【运行策略】策略结果：\n{message}")
    else:
        logger.info("【运行策略】当前市场无符合策略的合约交易机会")

# 每 1 分钟运行一次
def job():
    run_strategy()

if __name__ == "__main__":
    logger.info("OKX 合约交易策略机器人启动...")
    
    # 创建一个新线程执行调度
    def scheduler_thread():
        while True:
            schedule.run_pending()
            time.sleep(1)
    
    # 设置任务计划
    schedule.every(1).minutes.do(job)
    
    # 启动调度线程
    scheduler_thread = threading.Thread(target=scheduler_thread)
    scheduler_thread.start()

    # 主线程继续执行其他任务
    while True:
        time.sleep(10)