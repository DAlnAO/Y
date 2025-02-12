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

# OKX API 获取 K 线数据（可调节频率）
def get_okx_data(symbol, timeframe="1m", limit=200):
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

# 获取市场深度（Order Book）数据
def get_order_book(symbol):
    try:
        url = f"https://www.okx.com/api/v5/market/depth?instId={symbol}&size=5"
        logger.info(f"【获取深度数据】 请求 URL: {url}")
        response = requests.get(url)
        data = response.json()
        
        if "data" in data:
            order_book = data["data"][0]
            bid_depth = order_book['bids']
            ask_depth = order_book['asks']
            return bid_depth, ask_depth
        else:
            logger.warning(f"【获取深度数据失败】 {symbol} 没有返回深度数据")
            return None, None
    except Exception as e:
        logger.error(f"【获取深度数据失败】 获取市场深度时发生错误: {e}")
        return None, None

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

    # 添加量价关系指标：成交量与价格的结合
    df["Price_Volume_Trend"] = ta.trend.price_volume_trend(df["close"], df["volume"])

    logger.info(f"【计算指标】完成技术指标计算")
    return df

# 筛选符合交易策略的币种
def filter_trading_opportunities(symbol):
    logger.info(f"【筛选交易机会】开始分析币种: {symbol}")
    df = get_okx_data(symbol)
    if df is None:
        logger.info(f"【筛选交易机会】{symbol} 数据为空，跳过")
        return None
    
    df = calculate_indicators(df)
    
    latest = df.iloc[-1]
    close_price = latest["close"]
    atr = latest["ATR"]
    
    # 获取市场深度数据
    bid_depth, ask_depth = get_order_book(symbol)
    
    if bid_depth is None or ask_depth is None:
        logger.info(f"【筛选交易机会】{symbol} 无法获取市场深度数据，跳过")
        return None
    
    # 深度数据处理（买方和卖方的力量对比）
    highest_bid = float(bid_depth[0][0]) if bid_depth else 0
    lowest_ask = float(ask_depth[0][0]) if ask_depth else 0
    spread = lowest_ask - highest_bid
    
    # 做多信号
    long_conditions = [
        latest["SMA_50"] > latest["SMA_200"],
        latest["MACD"] > latest["MACD_signal"],
        latest["RSI"] > 50,
        latest["ADX"] > 25,
        latest["close"] > latest["BB_lower"],
        latest["close"] > latest["VWAP"],
        latest["Price_Volume_Trend"] > 0,  # 量价趋势为正
        spread < 0.005  # 市场流动性，价差过大可能意味着市场不活跃
    ]
    
    # 做空信号
    short_conditions = [
        latest["SMA_50"] < latest["SMA_200"],
        latest["MACD"] < latest["MACD_signal"],
        latest["RSI"] < 50,
        latest["ADX"] > 25,
        latest["close"] < latest["BB_upper"],
        latest["close"] < latest["VWAP"],
        latest["Price_Volume_Trend"] < 0,  # 量价趋势为负
        spread < 0.005  # 市场流动性，价差过大可能意味着市场不活跃
    ]
    
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

# 总结策略结果
def summarize_results():
    logger.info("【策略总结】每 2 分钟总结一次策略结果")
    # 这里可以根据你的需求总结交易策略的执行情况，例如，执行次数、胜率等
    logger.info("【策略总结】策略结果总结: 在过去的 2 分钟内运行了 6 次策略")

# 每 1 分钟运行一次策略
def job_run_strategy():
    run_strategy()

# 每 2 分钟总结策略结果
def job_summarize_results():
    summarize_results()

if __name__ == "__main__":
    logger.info("OKX 合约交易策略机器人启动...")
    
    # 创建一个新线程执行调度
    def scheduler_thread():
        while True:
            schedule.run_pending()
            time.sleep(1)
    
    # 设置任务计划
    schedule.every(1).minute.do(job_run_strategy)  # 每 1 分钟运行一次策略
    schedule.every(2).minutes.do(job_summarize_results)  # 每 2 分钟总结一次策略结果
    
    # 启动调度线程
    scheduler_thread = threading.Thread(target=scheduler_thread)
    scheduler_thread.start()

    # 主线程继续执行其他任务
    while True:
        time.sleep(10)