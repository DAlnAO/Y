import ccxt
import gym
import numpy as np
import pandas as pd
import time
import logging
import threading
from stable_baselines3 import PPO
from collections import deque
from datetime import datetime
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

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
symbols = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP", "ADA-USDT-SWAP", "BNB-USDT-SWAP"]
risk_percentage = 10  # 风险百分比（每次交易的最大风险）
max_position_percentage = 50  # 最大仓位占比
trading_frequency = 300  # 每5分钟交易一次
training_interval = 1800  # 每30分钟继续训练一次
stop_loss_percentage = 5  # 止损百分比
take_profit_percentage = 10  # 止盈百分比
max_drawdown_percentage = 20  # 最大回撤百分比

# 交易记录列表
trade_history = []
current_balance = 1000  # 初始账户余额
initial_balance = current_balance  # 用于回撤监控

# ✅ 获取账户余额
def get_balance():
    balance = exchange.fetch_balance()
    available_balance = balance['total']['USDT'] - balance['used']['USDT']  # 可用余额
    occupied_balance = balance['used']['USDT']  # 占用资金
    logging.info(f"账户信息: 当前余额: {balance['total']['USDT']}，占用资金: {occupied_balance}，可用余额: {available_balance}")
    return available_balance, occupied_balance

# ✅ 获取市场数据
def get_market_data(symbol, timeframes=['5m'], limit=500):
    try:
        market_data = {}
        for tf in timeframes:
            ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['ma5'] = df['close'].rolling(5).mean()
            df['ma15'] = df['close'].rolling(15).mean()
            df['atr'] = df['high'].rolling(14).max() - df['low'].rolling(14).min()
            df['rsi'] = 100 - (100 / (1 + df['close'].pct_change().rolling(14).mean()))
            df = df.dropna()
            market_data[tf] = df

            # 记录市场数据
            logging.info(f"市场数据 ({symbol} - {tf}): 最近数据点 - MA5: {df['ma5'].iloc[-1]}, MA15: {df['ma15'].iloc[-1]}, ATR: {df['atr'].iloc[-1]}, RSI: {df['rsi'].iloc[-1]}")
        
        return market_data
    except Exception as e:
        logging.error(f"⚠️ 获取市场数据失败: {symbol} | {e}")
        return None

# ✅ 计算根据ATR调整后的仓位
def calculate_position_size(usdt_balance, max_position_percentage, symbol):
    market_data = get_market_data(symbol)
    if not market_data:
        return 0
    
    current_price = market_data['5m']['close'].iloc[-1]
    atr = market_data['5m']['atr'].iloc[-1]
    
    # 基于ATR动态调整仓位
    volatility_factor = max(1, atr / current_price)  # ATR越大，波动性越高，仓位越小
    available_balance = usdt_balance * (max_position_percentage / 100) * (1 / volatility_factor)

    logging.info(f"计算仓位: {symbol} 基于ATR调整后的仓位为 {available_balance} USDT (当前价格: {current_price}, ATR: {atr})")
    
    return available_balance

# ✅ 检查最大回撤
def check_max_drawdown(current_balance, initial_balance):
    drawdown = (current_balance - initial_balance) / initial_balance * 100
    if drawdown <= -max_drawdown_percentage:
        logging.warning(f"⚠️ 最大回撤超过阈值 {max_drawdown_percentage}%，暂停交易")
        return True
    return False

# ✅ 执行交易
def execute_trade(symbol, action, usdt_balance):
    try:
        # 获取当前账户余额信息
        available_balance, occupied_balance = get_balance()
        
        # 计算仓位
        position_size = calculate_position_size(available_balance, max_position_percentage, symbol)
        
        # 记录交易前的账户信息
        logging.info(f"交易前账户信息: 当前余额 = {usdt_balance}, 可用余额 = {available_balance}, 占用资金 = {occupied_balance}")
        
        # 执行买入/卖出操作
        if action == "buy" and position_size > 0:
            ticker = exchange.fetch_ticker(symbol)
            price = ticker['ask']  # 获取买入价格（买一价）
            limit_price = price * 1.01  # 限价单稍微高于市场价格，减少滑点
            exchange.create_limit_buy_order(symbol, position_size, limit_price)
            logging.info(f"✅ 交易成功: {action.upper()} {position_size} {symbol} 限价单价格: {limit_price}")
        
        elif action == "sell" and position_size > 0:
            ticker = exchange.fetch_ticker(symbol)
            price = ticker['bid']  # 获取卖出价格（卖一价）
            limit_price = price * 0.99  # 限价单稍微低于市场价格，减少滑点
            exchange.create_limit_sell_order(symbol, position_size, limit_price)
            logging.info(f"✅ 交易成功: {action.upper()} {position_size} {symbol} 限价单价格: {limit_price}")
        
        # 记录交易
        trade_info = {
            "symbol": symbol,
            "action": action,
            "position_size": position_size,
            "price": exchange.fetch_ticker(symbol)['last'],
            "timestamp": time.time()
        }
        trade_history.append(trade_info)
        
        # 记录交易后的账户信息
        available_balance, occupied_balance = get_balance()
        logging.info(f"交易后账户信息: 当前余额 = {usdt_balance}, 可用余额 = {available_balance}, 占用资金 = {occupied_balance}")
    
    except Exception as e:
        logging.error(f"⚠️ 交易失败: {e}")

# ✅ 执行交易策略
def trade():
    for symbol in symbols:
        action = get_trade_signal(symbol)
        execute_trade(symbol, action, current_balance)

# ✅ 获取交易信号（PPO）
def get_trade_signal(symbol):
    try:
        ppo_model = PPO.load(f"ppo_trading_agent_{symbol}")
        data = get_market_data(symbol, timeframes=['5m'])
        if not data:
            logging.info(f"⚠️ 无法获取市场数据，无法生成交易信号")
            return "hold"

        df = data['5m']
        state = np.array([
            df['ma5'].iloc[-1],
            df['ma15'].iloc[-1],
            df['atr'].iloc[-1],
            df['rsi'].iloc[-1]
        ])

        # 记录状态输入
        logging.info(f"交易信号生成: {symbol} 当前状态 - MA5: {df['ma5'].iloc[-1]}, MA15: {df['ma15'].iloc[-1]}, ATR: {df['atr'].iloc[-1]}, RSI: {df['rsi'].iloc[-1]}")
        
        ppo_action, _ = ppo_model.predict(state)
        action = "buy" if ppo_action == 1 else "sell"
        logging.info(f"交易信号: {symbol} 推荐操作: {action.upper()}")
        return action
    except Exception as e:
        logging.error(f"⚠️ 获取交易信号失败: {e}")
        return "hold"

# ✅ 启动交易与训练
if __name__ == "__main__":
    while True:
        # 每5分钟反馈一次市场数据、信号及仓位等
        start_time = time.time()

        if check_max_drawdown(current_balance, initial_balance):
            break
        trade()
        
        # 每5分钟执行一次
        time.sleep(trading_frequency - (time.time() - start_time))