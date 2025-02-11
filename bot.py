import os
import ccxt
import gym
import numpy as np
import pandas as pd
import time
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

# ✅ 统一日志文件
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# ✅ OKX API 配置
exchange = ccxt.okx({
    'apiKey': "your_api_key",
    'secret': "your_secret",
    'password': "your_password",
    'options': {'defaultType': 'swap'},
})

# ✅ 交易参数
symbols = ["BTC-USDT-SWAP", "ETH-USDT-SWAP"]
risk_percentage = 10  
max_position_percentage = 50  
trading_frequency = 300  
stop_loss_percentage = 5  
take_profit_percentage = 10  

# ✅ **获取市场数据（多时间框架 + 更多特征）**
def get_market_data(symbol, timeframes=['5m', '15m', '1h'], limit=500):
    try:
        market_data = {}
        for tf in timeframes:
            ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # 技术指标计算
            df['ma5'] = df['close'].rolling(5).mean()
            df['ma15'] = df['close'].rolling(15).mean()
            df['atr'] = df['high'].rolling(14).max() - df['low'].rolling(14).min()
            df['rsi'] = 100 - (100 / (1 + df['close'].pct_change().rolling(14).mean()))
            df['vol_ma'] = df['volume'].rolling(10).mean()  # 成交量均线
            df['bollinger_up'] = df['close'].rolling(20).mean() + (df['close'].rolling(20).std() * 2)
            df['bollinger_down'] = df['close'].rolling(20).mean() - (df['close'].rolling(20).std() * 2)

            df = df.dropna()
            if not df.empty:
                market_data[tf] = df
        
        return market_data if market_data else None
    except Exception as e:
        logging.warning(f"⚠️ 获取市场数据失败: {symbol} | {e}")
        return None

# ✅ **强化学习环境（增加更多特征 + 多时间框架）**
class TradingEnv(gym.Env):
    def __init__(self, symbol):
        super(TradingEnv, self).__init__()
        self.symbol = symbol
        self.action_space = gym.spaces.Discrete(3)  # 0: 持有, 1: 买入, 2: 卖出
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.current_step = 0
        self.data = None

    def reset(self):
        self.current_step = 0
        self.data = get_market_data(self.symbol)
        if self.data is None or '5m' not in self.data or self.data['5m'].empty:
            return np.zeros(8) 

        self.state = np.array([
            self.data['5m']['ma5'].iloc[0],
            self.data['5m']['ma15'].iloc[0],
            self.data['15m']['ma5'].iloc[0],
            self.data['15m']['ma15'].iloc[0],
            self.data['5m']['atr'].iloc[0],
            self.data['5m']['rsi'].iloc[0],
            self.data['5m']['bollinger_up'].iloc[0],
            self.data['5m']['bollinger_down'].iloc[0]
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
    model.learn(total_timesteps=100000)
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
        df['rsi'].iloc[-1],
        df['bollinger_up'].iloc[-1],
        df['bollinger_down'].iloc[-1]
    ])

    ppo_action, _ = ppo_model.predict(state)

    if ppo_action == 1:
        return "buy"
    elif ppo_action == 2:
        return "sell"
    else:
        return "hold"

# ✅ **执行交易**
def execute_trade(symbol, action):
    try:
        balance = exchange.fetch_balance()
        usdt_balance = balance['total']['USDT']
        last_price = exchange.fetch_ticker(symbol)['last']

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