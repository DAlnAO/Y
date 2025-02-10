import os
import ccxt
import pandas as pd
import numpy as np
import time
import logging
from stable_baselines3 import SAC
from sklearn.preprocessing import MinMaxScaler

# ✅ 统一日志文件
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# ✅ OKX API 配置
exchange = ccxt.okx({
    'apiKey': "你的 API Key",
    'secret': "你的 API Secret",
    'password': "你的 API Password",
    'options': {'defaultType': 'swap'},
})

# ✅ 交易参数
risk_percentage = 10  
max_drawdown = 15
min_leverage = 5
max_leverage = 125
trade_history_file = "trade_history.csv"
symbols = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"]
training_interval = 86400  # 每 24 小时训练一次
training_count = 0  
consecutive_losses = 0  # 记录连续亏损次数
trading_frequency = 300  # 初始交易频率（5分钟）

# ✅ 获取市场数据
def get_market_data(symbol, timeframes=['5m', '1h', '1d'], limit=500):
    market_data = {}
    try:
        for tf in timeframes:
            ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # 计算技术指标
            df['ma5'] = df['close'].rolling(window=5).mean()
            df['ma15'] = df['close'].rolling(window=15).mean()
            df['rsi'] = 100 - (100 / (1 + df['close'].pct_change().rolling(14).mean()))
            df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
            df['atr'] = df['high'].rolling(14).max() - df['low'].rolling(14).min()
            df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
            df['boll_upper'] = df['close'].rolling(20).mean() + (df['close'].rolling(20).std() * 2)
            df['boll_lower'] = df['close'].rolling(20).mean() - (df['close'].rolling(20).std() * 2)
            df['cci'] = (df['close'] - df['close'].rolling(20).mean()) / (0.015 * df['close'].rolling(20).std())
            df['mfi'] = 100 - (100 / (1 + df['volume'].rolling(14).mean() / df['volume'].rolling(14).std()))
            df['adx'] = df['atr'].diff().abs().rolling(14).mean()

            df = df.dropna()
            market_data[tf] = df
        return market_data
    except Exception as e:
        logging.error(f"⚠️ 获取市场数据失败: {e}")
        return None

# ✅ 获取交易信号
def get_trade_signal(symbol):
    data = get_market_data(symbol, timeframes=['5m', '1h', '1d'])
    if not data:
        return "hold"

    short_term, mid_term, long_term = data['5m'], data['1h'], data['1d']
    
    short_signal = "buy" if short_term['ma5'].iloc[-1] > short_term['ma15'].iloc[-1] else "sell"
    mid_signal = "buy" if mid_term['ma5'].iloc[-1] > mid_term['ma15'].iloc[-1] else "sell"
    long_signal = "buy" if long_term['ma5'].iloc[-1] > long_term['ma15'].iloc[-1] else "sell"

    adx = short_term['adx'].iloc[-1]
    cci = short_term['cci'].iloc[-1]
    mfi = short_term['mfi'].iloc[-1]

    if adx < 20 or (mfi > 80 and short_signal == "buy") or (mfi < 20 and short_signal == "sell"):
        return "hold"

    final_signal = short_signal if short_signal == mid_signal == long_signal else "hold"
    return final_signal

# ✅ 计算动态止盈止损
def calculate_sl_tp(symbol, entry_price):
    df = get_market_data(symbol)['5m']
    atr = df['atr'].iloc[-1]
    stop_loss = entry_price - (atr * 1.5)
    take_profit = entry_price + (atr * 3)
    return stop_loss, take_profit

# ✅ 执行交易
def execute_trade(symbol, action, usdt_balance):
    global consecutive_losses
    try:
        leverage = get_dynamic_leverage(symbol)
        position_size = (usdt_balance * (risk_percentage / 100)) / leverage
        stop_loss, take_profit = calculate_sl_tp(symbol, get_market_data(symbol)['5m']['close'].iloc[-1])

        exchange.set_leverage(leverage, symbol, params={"mgnMode": "isolated"})
        order = exchange.create_market_order(symbol, action, position_size)
        logging.info(f"✅ 交易成功: {action.upper()} {position_size} 张 {symbol} | 杠杆: {leverage}x | 止损: {stop_loss} | 止盈: {take_profit}")

        consecutive_losses = 0  
    except Exception as e:
        logging.error(f"⚠️ 交易失败: {e}")
        consecutive_losses += 1

# ✅ 交易机器人
def trading_bot():
    global training_count, trading_frequency
    last_training_time = time.time()

    while True:
        try:
            balance = exchange.fetch_balance()
            usdt_balance = balance['total'].get('USDT', 0)

            for symbol in symbols:
                signal = get_trade_signal(symbol)
                if signal in ["buy", "sell"]:
                    execute_trade(symbol, signal, usdt_balance)

            logging.info(f"💰 账户余额: {usdt_balance} USDT | 交易频率: {trading_frequency} 秒")
            time.sleep(trading_frequency) 
        
        except Exception as e:
            logging.error(f"⚠️ 交易循环错误: {e}")
            time.sleep(trading_frequency)

# ✅ 启动机器人
trading_bot()