import os
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
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

# ✅ 统一日志文件
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# ✅ OKX API 配置
exchange = ccxt.okx({
    'apiKey': "your_api_key",
    'secret': "your_secret_key",
    'password': "your_api_password",
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
            logging.info(f"市场数据 ({symbol} - {tf}): MA5: {df['ma5'].iloc[-1]}, MA15: {df['ma15'].iloc[-1]}, ATR: {df['atr'].iloc[-1]}, RSI: {df['rsi'].iloc[-1]}")
        
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
    
    volatility_factor = max(1, atr / current_price)  
    available_balance = usdt_balance * (max_position_percentage / 100) * (1 / volatility_factor)

    logging.info(f"计算仓位: {symbol} ATR调整后仓位 = {available_balance} USDT (当前价格: {current_price}, ATR: {atr})")
    
    return available_balance

# ✅ 检查最大回撤
def check_max_drawdown(current_balance, initial_balance):
    drawdown = (current_balance - initial_balance) / initial_balance * 100
    if drawdown <= -max_drawdown_percentage:
        logging.warning(f"⚠️ 最大回撤超过阈值 {max_drawdown_percentage}%，暂停交易")
        return True
    return False

# ✅ 训练模型
def train_model(symbol):
    class TradingEnv(gym.Env):
        def __init__(self):
            super(TradingEnv, self).__init__()
            self.action_space = gym.spaces.Discrete(2)
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        def reset(self):
            self.current_step = 0
            self.data = get_market_data(symbol)
            self.state = np.array([
                self.data['5m']['ma5'].iloc[0],
                self.data['5m']['ma15'].iloc[0],
                self.data['5m']['atr'].iloc[0],
                self.data['5m']['rsi'].iloc[0]
            ])
            return self.state

        def step(self, action):
            self.current_step += 1
            reward = 0  
            if self.current_step >= len(self.data['5m']):
                done = True
            else:
                done = False
            return self.state, reward, done, {}

    env = DummyVecEnv([lambda: TradingEnv()])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save(f"ppo_trading_agent_{symbol}")
    logging.info(f"✅ 训练完成: {symbol}")

# ✅ 获取交易信号（PPO）
def get_trade_signal(symbol):
    model_file = f"ppo_trading_agent_{symbol}.zip"
    
    if not os.path.exists(model_file):
        logging.info(f"⚠️ 未找到模型 {model_file}，正在训练...")
        train_model(symbol)
    
    ppo_model = PPO.load(model_file)
    data = get_market_data(symbol, timeframes=['5m'])
    if not data:
        return "hold"

    df = data['5m']
    state = np.array([
        df['ma5'].iloc[-1],
        df['ma15'].iloc[-1],
        df['atr'].iloc[-1],
        df['rsi'].iloc[-1]
    ])

    ppo_action, _ = ppo_model.predict(state)
    return "buy" if ppo_action == 1 else "sell"

# ✅ 执行交易
def execute_trade(symbol, action, usdt_balance):
    try:
        available_balance, occupied_balance = get_balance()
        position_size = calculate_position_size(available_balance, max_position_percentage, symbol)

        if action == "buy" and position_size > 0:
            exchange.create_market_buy_order(symbol, position_size)
            logging.info(f"✅ 交易成功: {action.upper()} {position_size} {symbol}")
        
        elif action == "sell" and position_size > 0:
            exchange.create_market_sell_order(symbol, position_size)
            logging.info(f"✅ 交易成功: {action.upper()} {position_size} {symbol}")
    except Exception as e:
        logging.error(f"⚠️ 交易失败: {e}")

# ✅ 启动交易
if __name__ == "__main__":
    while True:
        if check_max_drawdown(current_balance, initial_balance):
            break
        for symbol in symbols:
            action = get_trade_signal(symbol)
            execute_trade(symbol, action, current_balance)
        time.sleep(trading_frequency)