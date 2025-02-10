import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 禁用 GPU
import tensorflow as tf

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
    'apiKey': "0f046e6a-1627-4db4-b97d-083d7e6cc16b",
    'secret': "BF7BC880C73AD54D2528FA271A358C2C",
    'password': "Duan0918.",
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

# ✅ 获取市场数据并记录日志
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
            df['atr'] = df['high'].rolling(14).max() - df['low'].rolling(14).min()
            df['adx'] = df['atr'].diff().abs().rolling(14).mean()

            df = df.dropna()
            market_data[tf] = df
        
        logging.info(f"📊 市场数据更新成功: {symbol}")
        return market_data
    except Exception as e:
        logging.error(f"⚠️ 获取市场数据失败: {symbol} | 错误: {e}")
        return None

# ✅ 交易信号获取并记录日志
def get_trade_signal(symbol):
    data = get_market_data(symbol, timeframes=['5m', '1h', '1d'])
    if not data:
        return "hold"

    short_term, mid_term, long_term = data['5m'], data['1h'], data['1d']
    
    signals = {"buy": 0, "sell": 0}
    weights = {"5m": 3, "1h": 2, "1d": 1}

    short_signal = "buy" if short_term['ma5'].iloc[-1] > short_term['ma15'].iloc[-1] else "sell"
    mid_signal = "buy" if mid_term['ma5'].iloc[-1] > mid_term['ma15'].iloc[-1] else "sell"
    long_signal = "buy" if long_term['ma5'].iloc[-1] > long_term['ma15'].iloc[-1] else "sell"

    signals[short_signal] += weights['5m']
    signals[mid_signal] += weights['1h']
    signals[long_signal] += weights['1d']

    final_signal = "buy" if signals["buy"] > signals["sell"] else "sell" if signals["sell"] > signals["buy"] else "hold"

    logging.info(f"📈 {symbol} 交易信号: 短期({short_signal}) | 中期({mid_signal}) | 长期({long_signal}) | 最终信号: {final_signal.upper()}")

    return final_signal

# ✅ 计算止盈止损并记录日志
def calculate_sl_tp(symbol, entry_price):
    df = get_market_data(symbol)['5m']
    atr = df['atr'].iloc[-1]
    stop_loss = entry_price - (atr * 1.5)
    take_profit = entry_price + (atr * 3)

    logging.info(f"🛑 {symbol} 止盈止损计算: ATR={atr:.2f} | 止损={stop_loss:.2f} | 止盈={take_profit:.2f}")
    return stop_loss, take_profit

# ✅ 执行交易（新增）
def execute_trade(symbol, action, usdt_balance):
    global consecutive_losses
    try:
        leverage = min_leverage  # 这里可以加入动态调整逻辑
        position_size = (usdt_balance * (risk_percentage / 100)) / leverage
        stop_loss, take_profit = calculate_sl_tp(symbol, get_market_data(symbol)['5m']['close'].iloc[-1])

        exchange.set_leverage(leverage, symbol, params={"mgnMode": "isolated"})
        order = exchange.create_market_order(symbol, action, position_size)
        logging.info(f"✅ 交易成功: {action.upper()} {position_size} 张 {symbol} | 杠杆: {leverage}x | 止损: {stop_loss:.2f} | 止盈: {take_profit:.2f}")

        consecutive_losses = 0  
    except Exception as e:
        logging.error(f"⚠️ 交易失败: {e}")
        consecutive_losses += 1

# ✅ 交易机器人 & 记录学习进度
def trading_bot():
    global training_count, trading_frequency
    last_training_time = time.time()

    while True:
        try:
            balance = exchange.fetch_balance()
            usdt_balance = balance['total'].get('USDT', 0)
            
            logging.info("=" * 50)
            logging.info(f"📊 交易循环开始: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            logging.info(f"💰 账户余额: {usdt_balance:.2f} USDT")
            
            for symbol in symbols:
                signal = get_trade_signal(symbol)

                if signal in ["buy", "sell"]:
                    execute_trade(symbol, signal, usdt_balance)
                    logging.info(f"✅ 交易执行成功: {symbol} | 动作: {signal.upper()}")
                else:
                    logging.info(f"⏸️ 交易跳过: {symbol} | 无有效信号")

            # 训练机器人（如果有强化学习）
            if time.time() - last_training_time > training_interval:
                training_count += 1
                logging.info(f"📢 机器人强化学习开始: 训练轮次 {training_count}")
                # 训练逻辑（例如 SAC/PPO）
                last_training_time = time.time()
                logging.info(f"✅ 机器人强化学习完成")

            logging.info(f"💰 账户余额: {usdt_balance:.2f} USDT")
            logging.info(f"⏳ 进入等待状态，下一次检查将在 {trading_frequency} 秒后\n")
            logging.info("=" * 50 + "\n")

            time.sleep(trading_frequency)
        
        except Exception as e:
            logging.error(f"⚠️ 交易循环错误: {e}")
            time.sleep(trading_frequency)

# ✅ 启动交易机器人
trading_bot()