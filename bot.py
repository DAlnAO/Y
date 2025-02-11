import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import ccxt
import gym
import numpy as np
import pandas as pd
import time
import logging
import pickle
import threading
from stable_baselines3 import PPO
from sklearn.ensemble import RandomForestClassifier
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
symbols = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"]
risk_percentage = 10  
max_position_percentage = 50  
trading_frequency = 300  
training_interval = 86400  

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
        return market_data
    except Exception as e:
        logging.error(f"⚠️ 获取市场数据失败: {symbol} | {e}")
        return None

# ✅ PPO 强化学习交易环境
class TradingEnv(gym.Env):
    def __init__(self, symbol):
        super(TradingEnv, self).__init__()
        self.symbol = symbol
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.market_data = get_market_data(symbol)['5m']
        self.current_step = 20
        self.balance = 1000
        self.position = 0

    def step(self, action):
        prev_price = self.market_data['close'].iloc[self.current_step - 1]
        current_price = self.market_data['close'].iloc[self.current_step]
        reward = 0

        if action == 1:
            self.position = self.balance / current_price
            self.balance = 0
        elif action == 2:
            self.balance = self.position * current_price
            self.position = 0
            reward = self.balance - 1000

        self.current_step += 1
        done = self.current_step >= len(self.market_data) - 1
        state = self._get_state()

        return state, reward, done, {}

    def reset(self):
        self.current_step = 20
        self.balance = 1000
        self.position = 0
        return self._get_state()

    def _get_state(self):
        return np.array([
            self.market_data['ma5'].iloc[self.current_step],
            self.market_data['ma15'].iloc[self.current_step],
            self.market_data['atr'].iloc[self.current_step],
            self.market_data['rsi'].iloc[self.current_step]
        ])

# ✅ 训练 PPO 交易代理
def train_ppo_agent(symbol):
    env = TradingEnv(symbol)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save(f"ppo_trading_agent_{symbol}")
    logging.info(f"✅ {symbol} PPO 训练完成")

# ✅ 训练 & 自动更新 RF 模型
def retrain_rf_model(symbol):
    while True:
        data = get_market_data(symbol, timeframes=['5m'])
        if data is None:
            time.sleep(86400)
            continue

        df = data['5m']
        df['future_return'] = df['close'].shift(-1) / df['close'] - 1
        df['label'] = (df['future_return'] > 0).astype(int)
        features = ['ma5', 'ma15', 'atr', 'rsi']
        df = df.dropna()

        if df.empty:
            time.sleep(86400)
            continue

        X, y = df[features], df['label']
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)

        with open(f"rf_model_{symbol}.pkl", "wb") as f:
            pickle.dump(rf, f)
        
        logging.info(f"✅ {symbol} RF 模型已更新")
        time.sleep(86400)

# ✅ 获取交易信号
def get_trade_signal_rf(symbol, model):
    data = get_market_data(symbol, timeframes=['5m'])
    if not data:
        return "hold"

    df = data['5m']
    features = ['ma5', 'ma15', 'atr', 'rsi']
    prediction = model.predict(df[features].iloc[-1:].values)[0]

    return "buy" if prediction == 1 else "sell"

# ✅ 交易执行
def execute_trade(symbol, action, usdt_balance):
    try:
        position_size = (usdt_balance * (risk_percentage / 100))
        exchange.create_market_order(symbol, action, position_size)
        logging.info(f"✅ 交易成功: {action.upper()} {position_size} {symbol}")
    except Exception as e:
        logging.error(f"⚠️ 交易失败: {e}")

# ✅ 交易机器人
def trading_bot():
    for symbol in symbols:
        threading.Thread(target=retrain_rf_model, args=(symbol,), daemon=True).start()

    while True:
        try:
            usdt_balance = exchange.fetch_balance()['total'].get('USDT', 0)
            for symbol in symbols:
                try:
                    with open(f"rf_model_{symbol}.pkl", "rb") as f:
                        rf_model = pickle.load(f)
                except FileNotFoundError:
                    logging.warning(f"⚠️ {symbol} 还没有训练好的 RF 模型")
                    continue

                signal = get_trade_signal_rf(symbol, rf_model)
                if signal in ["buy", "sell"]:
                    execute_trade(symbol, signal, usdt_balance)

            time.sleep(trading_frequency)
        except Exception as e:
            logging.error(f"⚠️ 交易循环错误: {e}")
            time.sleep(trading_frequency)

# ✅ 启动交易机器人
trading_bot()