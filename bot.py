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
    'apiKey': "0f046e6a-1627-4db4-b97d-083d7e6cc16b",
    'secret': "BF7BC880C73AD54D2528FA271A358C2C",
    'password': "Duan0918.",
    'options': {'defaultType': 'swap'},
})

# ✅ 交易参数
symbols = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP", "ADA-USDT-SWAP", "BNB-USDT-SWAP"]
risk_percentage = 10  
max_position_percentage = 50  
trading_frequency = 300  
training_interval = 1800  
stop_loss_percentage = 5  
take_profit_percentage = 10  
max_drawdown_percentage = 20  

# 交易记录列表
trade_history = []
current_balance = 1000  
initial_balance = current_balance  

# ✅ **获取市场数据（带重试机制）**
def get_market_data(symbol, timeframes=['5m'], limit=500, retries=3, delay=5):
    for attempt in range(retries):
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

                if not df.empty:
                    market_data[tf] = df

                    logging.info(f"市场数据 ({symbol} - {tf}): MA5: {df['ma5'].iloc[-1]}, MA15: {df['ma15'].iloc[-1]}, ATR: {df['atr'].iloc[-1]}, RSI: {df['rsi'].iloc[-1]}")
            
            if market_data:
                return market_data
        except Exception as e:
            logging.warning(f"⚠️ 第 {attempt+1}/{retries} 次尝试获取市场数据失败: {symbol} | {e}")
            time.sleep(delay)
    
    logging.error(f"❌ 获取市场数据失败: {symbol}，返回空数据")
    return None

# ✅ **强化学习环境**
class TradingEnv(gym.Env):
    def __init__(self, symbol):
        super(TradingEnv, self).__init__()
        self.symbol = symbol
        self.action_space = gym.spaces.Discrete(2)  
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.current_step = 0
        self.data = None

    def reset(self):
        self.current_step = 0
        self.data = get_market_data(self.symbol)

        if self.data is None or '5m' not in self.data or self.data['5m'].empty:
            logging.warning(f"⚠️ {self.symbol} 没有市场数据，返回默认状态")
            return np.zeros(4) 

        self.state = np.array([
            self.data['5m']['ma5'].iloc[0],
            self.data['5m']['ma15'].iloc[0],
            self.data['5m']['atr'].iloc[0],
            self.data['5m']['rsi'].iloc[0]
        ])
        return self.state

    def step(self, action):
        self.current_step += 1
        if self.current_step >= len(self.data['5m']):
            return self.state, 0, True, {}
        return self.state, 0, False, {}

# ✅ **训练模型**
def train_model(symbol):
    logging.info(f"开始训练模型: {symbol}")

    env = DummyVecEnv([lambda: TradingEnv(symbol)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save(f"ppo_trading_agent_{symbol}")

    logging.info(f"✅ 训练完成: {symbol}")

# ✅ **获取交易信号**
def get_trade_signal(symbol):
    model_file = f"ppo_trading_agent_{symbol}.zip"
    
    if not os.path.exists(model_file):
        logging.info(f"⚠️ 模型 {model_file} 不存在，正在训练...")
        train_model(symbol)
    
    ppo_model = PPO.load(model_file)
    data = get_market_data(symbol, timeframes=['5m'])

    if not data or '5m' not in data or data['5m'].empty:
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

# ✅ **执行交易**
def execute_trade(symbol, action):
    try:
        balance = exchange.fetch_balance()
        usdt_balance = balance['total']['USDT']
        
        if action == "buy":
            exchange.create_market_buy_order(symbol, 1)
            logging.info(f"✅ {symbol} 买入成功")
        elif action == "sell":
            exchange.create_market_sell_order(symbol, 1)
            logging.info(f"✅ {symbol} 卖出成功")
    except Exception as e:
        logging.error(f"⚠️ 交易失败: {e}")

# ✅ **交易循环**
if __name__ == "__main__":
    while True:
        for symbol in symbols:
            action = get_trade_signal(symbol)
            if action in ["buy", "sell"]:
                execute_trade(symbol, action)
        time.sleep(trading_frequency)