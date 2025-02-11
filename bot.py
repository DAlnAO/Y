import os
import ccxt
import gym
import numpy as np
import pandas as pd
import time
import logging
import pickle
from dotenv import load_dotenv
from stable_baselines3 import PPO
from datetime import datetime
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ✅ 加载环境变量 (存储API Key安全方式)
load_dotenv()

# ✅ 日志记录
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# ✅ OKX API 配置 (从环境变量获取密钥)
exchange = ccxt.okx({
    'apiKey': os.getenv("0f046e6a-1627-4db4-b97d-083d7e6cc16b"),
    'secret': os.getenv("BF7BC880C73AD54D2528FA271A358C2C"),
    'password': os.getenv("Duan0918."),
    'options': {'defaultType': 'swap'},
})

# ✅ 交易参数
symbols = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP", "ADA-USDT-SWAP"]
update_interval = 1800  # 30分钟更新模型
risk_percentage = 0.1  # 每笔交易使用余额的10%

# ✅ 获取账户余额 (自动计算交易量)
def get_trade_amount(symbol):
    try:
        balance = exchange.fetch_balance()
        usdt_balance = balance.get('total', {}).get('USDT', 0)  # 避免KeyError
        ticker = exchange.fetch_ticker(symbol)
        price = ticker['last']

        trade_amount = (usdt_balance * risk_percentage) / price
        return round(trade_amount, 4) if trade_amount > 0 else 0.01
    except Exception as e:
        logging.error(f"❌ 获取交易金额失败: {symbol}, 错误: {e}")
        return 0.01  # 默认交易量

# ✅ 获取市场数据
def get_market_data(symbol, timeframes=['5m'], limit=500):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframes[0], limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # 计算技术指标
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma15'] = df['close'].rolling(15).mean()
        df['atr'] = df['high'].rolling(14).max() - df['low'].rolling(14).min()
        df['rsi'] = 100 - (100 / (1 + df['close'].pct_change().rolling(14).mean()))
        df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        df['bollinger_up'] = df['close'].rolling(20).mean() + 2 * df['close'].rolling(20).std()
        df['bollinger_down'] = df['close'].rolling(20).mean() - 2 * df['close'].rolling(20).std()
        df = df.dropna()  # 删除NaN值

        return df
    except Exception as e:
        logging.error(f"❌ 获取市场数据失败: {symbol}, 错误: {e}")
        return None

# ✅ 强化学习环境
class TradingEnv(gym.Env):
    def __init__(self, symbol):
        super(TradingEnv, self).__init__()
        self.symbol = symbol
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        self.data = get_market_data(symbol)
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        self.data = get_market_data(self.symbol)
        return self.get_state()

    def get_state(self):
        df = self.data.iloc[self.current_step]
        return np.array([
            df['ma5'], df['ma15'], df['atr'], df['rsi'], df['macd'],
            df['bollinger_up'], df['bollinger_down']
        ], dtype=np.float32).reshape(1, -1)  # ✅ 确保shape (1, 7)

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        return self.get_state(), 0, done, {}

# ✅ 获取交易信号 (结合PPO)
def get_trade_signal(symbol):
    try:
        model = PPO.load(f"ppo_trading_agent_{symbol}.zip")
    except:
        logging.warning(f"⚠️ 未找到 PPO 模型 {symbol}, 跳过交易")
        return "hold"

    data = get_market_data(symbol)
    if data is None or data.empty:
        return "hold"

    state = data.iloc[-1][['ma5', 'ma15', 'atr', 'rsi', 'macd', 'bollinger_up', 'bollinger_down']].values
    state = np.expand_dims(state, axis=0)  # ✅ 确保 (1, 7)

    action, _ = model.predict(state)
    execute_trade(symbol, "buy" if action == 1 else "sell")

# ✅ 执行交易
def execute_trade(symbol, action):
    trade_amount = get_trade_amount(symbol)
    if trade_amount <= 0:
        logging.warning(f"⚠️ {symbol} 交易量过低, 跳过交易")
        return

    try:
        order = exchange.create_order(symbol, type="market", side=action, amount=trade_amount)
        logging.info(f"✅ {action.upper()} {symbol}, 数量: {trade_amount}, 订单ID: {order['id']}")
    except Exception as e:
        logging.error(f"❌ 交易失败: {symbol}, 错误: {e}")

# ✅ 交易循环
if __name__ == "__main__":
    last_update_time = time.time()

    while True:
        for symbol in symbols:
            get_trade_signal(symbol)

        if time.time() - last_update_time > update_interval:
            last_update_time = time.time()

        logging.info(f"⏳ 休眠 300 秒后继续交易")
        time.sleep(300)