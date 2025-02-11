import os
import ccxt
import gym
import numpy as np
import pandas as pd
import time
import logging
from stable_baselines3 import PPO
from datetime import datetime
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

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
symbols = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP", "ADA-USDT-SWAP", "BNB-USDT-SWAP", "XRP-USDT-SWAP", "LTC-USDT-SWAP"]
trading_frequency = 300  
update_interval = 1800  # 每 30 分钟更新 PPO
stop_loss_multiplier = 1.5  
take_profit_multiplier = 2.5  
max_drawdown_percentage = 20  

# ✅ 获取市场数据（增加 MACD, Bollinger Bands）
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
        df = df.dropna()

        return df
    except Exception as e:
        logging.error(f"❌ 获取市场数据失败: {symbol}，错误: {e}")
        return None

# ✅ 强化学习环境（确保观察空间和状态形状匹配）
class TradingEnv(gym.Env):
    def __init__(self, symbol):
        super(TradingEnv, self).__init__()
        self.symbol = symbol
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)  # 确保 shape=(7,)
        self.data = get_market_data(symbol)
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        self.data = get_market_data(self.symbol)
        return self.get_state()

    def get_state(self):
        df = self.data.iloc[self.current_step]
        return np.array([df['ma5'], df['ma15'], df['atr'], df['rsi'], df['macd'], df['bollinger_up'], df['bollinger_down']], dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        return self.get_state(), 0, done, {}

# ✅ 训练 PPO（确保 `VecNormalize` 形状正确）
def update_ppo_model(symbol):
    model_file = f"ppo_trading_agent_{symbol}.zip"

    if os.path.exists(model_file):
        model = PPO.load(model_file)
        logging.info(f"🔄 继续训练 PPO 模型: {symbol}")
    else:
        logging.warning(f"⚠️ {symbol} 没有现有 PPO 模型，重新训练")
        model = PPO("MlpPolicy", DummyVecEnv([lambda: TradingEnv(symbol)]), verbose=1)

    env = DummyVecEnv([lambda: TradingEnv(symbol)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, training=True)  # 确保 training=True

    model.learn(total_timesteps=5000)
    model.save(model_file)
    logging.info(f"✅ PPO 模型 {symbol} 更新完成")

# ✅ 获取交易信号（确保 `state` 形状正确）
def get_trade_signal(symbol):
    model_file = f"ppo_trading_agent_{symbol}.zip"
    if not os.path.exists(model_file):
        update_ppo_model(symbol)

    model = PPO.load(model_file)
    data = get_market_data(symbol)
    if data is None:
        return "hold"

    df = data.iloc[-1]
    state = np.array([df['ma5'], df['ma15'], df['atr'], df['rsi'], df['macd'], df['bollinger_up'], df['bollinger_down']], dtype=np.float32)
    
    # ✅ 重要：确保 `state` 形状为 (1, 7)，避免 PPO 形状错误
    state = state.reshape(1, -1)

    action, _ = model.predict(state)
    return "buy" if action == 1 else "sell"
# ✅ 交易循环
if __name__ == "__main__":
    last_update_time = time.time()

    while True:
        for symbol in symbols:
            action = get_trade_signal(symbol)
            if action in ["buy", "sell"]:
                logging.info(f"{symbol}: {action}")

        # 每 30 分钟更新 PPO 模型
        if time.time() - last_update_time > update_interval:
            logging.info("🔄 开始增量训练 PPO 模型...")
            for symbol in symbols:
                update_ppo_model(symbol)
            last_update_time = time.time()
            logging.info("✅ PPO 训练更新完成")

        time.sleep(trading_frequency)