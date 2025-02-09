import ccxt
import pandas as pd
import numpy as np
import talib
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from stable_baselines3 import DQN
import gym
from gym import spaces
import time
import logging

# **日志系统**
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# **OKX API 配置**
exchange = ccxt.okx({
    'apiKey': "你的API_KEY",
    'secret': "你的API_SECRET",
    'password': "你的API_PASSPHRASE",
    'options': {'defaultType': 'swap'},
})

# **数据采集**
def get_market_data(symbol='ETH-USDT-SWAP', timeframe='15m', limit=500):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# **添加技术指标**
def add_technical_indicators(df):
    df['ma5'] = talib.SMA(df['close'], timeperiod=5)
    df['ma15'] = talib.SMA(df['close'], timeperiod=15)
    df['ma50'] = talib.SMA(df['close'], timeperiod=50)
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['macd'], df['macd_signal'], _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    return df

# **交易环境（强化学习）**
class TradingEnv(gym.Env):
    def __init__(self, symbol='ETH-USDT-SWAP', timeframe='15m', lookback=50):
        super(TradingEnv, self).__init__()

        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe
        self.lookback = lookback
        self.data = self.get_market_data()
        self.current_step = lookback
        self.balance = 10000
        self.position = 0

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(lookback, 5), dtype=np.float32)

    def get_market_data(self):
        df = get_market_data(self.symbol, self.timeframe, limit=1000)
        return df

    def step(self, action):
        prev_price = self.data.iloc[self.current_step - 1]['close']
        current_price = self.data.iloc[self.current_step]['close']

        if action == 0 and self.position == 0:
            self.position = self.balance / current_price
            self.balance = 0
        elif action == 1 and self.position > 0:
            self.balance = self.position * current_price
            self.position = 0

        new_balance = self.balance + (self.position * current_price)
        reward = new_balance - self.balance
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        obs = self.data.iloc[self.current_step - self.lookback:self.current_step].values

        return obs, reward, done, {}

    def reset(self):
        self.current_step = self.lookback
        self.balance = 10000
        self.position = 0
        return self.data.iloc[self.current_step - self.lookback:self.current_step].values

# **训练 DQN 强化学习 Agent**
def train_rl_model():
    env = TradingEnv()
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)
    model.save("dqn_trading_model")

# **加载训练好的 DQN 模型**
model_dqn = DQN.load("dqn_trading_model")

# **获取交易信号**
def get_trade_signal():
    df = get_market_data('ETH-USDT-SWAP', '15m', 500)
    df = add_technical_indicators(df)

    # 机器学习预测
    X = df[['ma5', 'ma15', 'ma50', 'rsi', 'atr', 'macd']]
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(X[:-1], (df['close'].shift(-1) > df['close'])[:-1].astype(int))
    short_term_signal = xgb_model.predict(X[-1:])[0]

    # 强化学习决策
    env = TradingEnv()
    obs = env.reset()
    rl_action, _ = model_dqn.predict(obs)

    # 综合信号
    if short_term_signal == 1 and rl_action == 0:
        return "buy"
    elif short_term_signal == 0 and rl_action == 1:
        return "sell"
    else:
        return "hold"

# **交易执行**
def place_order(symbol, side, size):
    order = {
        "instId": symbol,
        "tdMode": "isolated",
        "side": side,
        "ordType": "market",
        "sz": str(size),
    }
    try:
        exchange.private_post_trade_order(order)
        print(f"✅ {side.upper()} {symbol} x {size}")
    except Exception as e:
        print(f"⚠️ 下单失败: {e}")

# **实盘交易机器人**
def trading_bot(symbol='ETH-USDT-SWAP'):
    balance = exchange.fetch_balance()['total']['USDT']
    position_size = round(balance * 0.1, 2)

    while True:
        signal = get_trade_signal()
        position = get_position(symbol)

        if signal == "buy" and not position:
            place_order(symbol, "buy", position_size)
        elif signal == "sell" and not position:
            place_order(symbol, "sell", position_size)
        elif position and signal == "hold":
            close_position(symbol)

        time.sleep(10)

# **启动 AI 交易机器人**
trading_bot()