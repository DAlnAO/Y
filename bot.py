import ccxt
import pandas as pd
import numpy as np
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from bayes_opt import BayesianOptimization
from sklearn.ensemble import RandomForestClassifier
from stable_baselines3 import PPO
import gym
from websocket import create_connection
import json

# ✅ 统一日志文件
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# ✅ OKX API 配置
exchange = ccxt.okx({
    'apiKey': "0f046e6a-1627-4db4-b97d-083d7e6cc16b",
    'secret': "BF7BC880C73AD54D2528FA271A358C2C",
    'password': "Duan0918.",
    'options': {'defaultType': 'swap'}

# ✅ 定义交易对
symbols = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"]  # 🔥 添加这行，防止 `NameError`

# ✅ 缓存市场数据（60 秒有效）
market_cache = {}
cache_expiration = 60  # 缓存 60 秒

# ✅ API 请求自动重试
def api_request_with_retry(request_func, *args, max_retries=3, delay=3):
    for attempt in range(max_retries):
        try:
            return request_func(*args)
        except Exception as e:
            logging.warning(f"⚠️ API 请求失败: {e}，重试 {attempt + 1}/{max_retries}")
            time.sleep(delay)
    logging.error("🚫 API 多次请求失败，跳过此操作")
    return None

# ✅ 批量获取市场数据
def get_batch_market_data(symbols):
    tickers = api_request_with_retry(exchange.fetch_tickers)
    return {symbol: tickers.get(symbol, {}) for symbol in symbols}

# ✅ 获取市场数据（缓存优化）
def get_market_data(symbol, timeframes=['5m'], limit=500):
    global market_cache
    current_time = time.time()

    if symbol in market_cache and (current_time - market_cache[symbol]['timestamp'] < cache_expiration):
        return market_cache[symbol]['data']

    try:
        market_data = {}
        for tf in timeframes:
            ohlcv = api_request_with_retry(exchange.fetch_ohlcv, symbol, tf, limit)
            if ohlcv is None:
                continue

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # 计算技术指标
            df['ma5'] = df['close'].rolling(5).mean()
            df['ma15'] = df['close'].rolling(15).mean()
            df['atr'] = df['high'].rolling(14).max() - df['low'].rolling(14).min()
            df['rsi'] = 100 - (100 / (1 + df['close'].diff().rolling(14).apply(lambda x: (x[x > 0].sum() / abs(x[x < 0].sum())) if abs(x[x < 0].sum()) > 0 else 0)))
            df = df.dropna()
            market_data[tf] = df

        market_cache[symbol] = {'data': market_data, 'timestamp': current_time}
        return market_data
    except Exception as e:
        logging.error(f"⚠️ 获取市场数据失败: {symbol} | 错误: {e}")
        return None

# ✅ 训练随机森林交易信号
def train_rf_model(symbol):
    data = get_market_data(symbol, timeframes=['5m'])
    if not data:
        return None

    df = data['5m']
    df['future_return'] = df['close'].shift(-1) / df['close'] - 1
    df['label'] = (df['future_return'] > 0).astype(int)

    features = ['ma5', 'ma15', 'atr', 'rsi']
    df = df.dropna()

    X, y = df[features], df['label']
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    logging.info(f"✅ {symbol} 随机森林模型训练完成")
    return rf

# ✅ 交易信号生成（随机森林）
def get_trade_signal_rf(symbol, model):
    data = get_market_data(symbol, timeframes=['5m'])
    if not data:
        return "hold"

    df = data['5m']
    features = ['ma5', 'ma15', 'atr', 'rsi']
    prediction = model.predict(df[features].iloc[-1:].values)[0]

    return "buy" if prediction == 1 else "sell"

# ✅ 计算动态仓位（市场波动率）
def calculate_dynamic_position(symbol, usdt_balance):
    market_data = get_market_data(symbol, timeframes=['5m'])
    if not market_data:
        return 0

    df = market_data['5m']
    atr = df['atr'].iloc[-1]

    base_risk = 0.05
    risk_adjusted = max(0.01, base_risk / (atr * 10))
    position_size = usdt_balance * risk_adjusted

    logging.info(f"📊 {symbol} 动态仓位: 波动率={atr:.4f}, 仓位比例={risk_adjusted:.2f}, 资金分配={position_size:.2f} USDT")
    return position_size

# ✅ 强化学习（PPO）优化交易策略
class TradingEnv(gym.Env):
    def __init__(self):
        super(TradingEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

    def step(self, action):
        reward = np.random.uniform(-1, 1)
        return np.random.randn(6), reward, False, {}

    def reset(self):
        return np.random.randn(6)

ppo_model = PPO("MlpPolicy", TradingEnv(), verbose=0)

# ✅ 交易执行
def execute_trade(symbol, action, usdt_balance):
    try:
        position_size = calculate_dynamic_position(symbol, usdt_balance)
        if position_size > 0:
            exchange.create_market_order(symbol, action, position_size)
            logging.info(f"✅ 交易成功: {action.upper()} {position_size} 张 {symbol}")
        else:
            logging.info(f"⚠️ {symbol} 仓位太小，跳过交易")
    except Exception as e:
        logging.error(f"⚠️ 交易失败: {e}")

# ✅ 交易机器人循环
def trading_bot():
    models = {symbol: train_rf_model(symbol) for symbol in symbols}

    while True:
        try:
            usdt_balance = exchange.fetch_balance()['total'].get('USDT', 0)

            for symbol in symbols:
                model = models.get(symbol)
                if model:
                    signal = get_trade_signal_rf(symbol, model)
                    if signal in ["buy", "sell"]:
                        execute_trade(symbol, signal, usdt_balance)

            time.sleep(300)
        except Exception as e:
            logging.error(f"⚠️ 交易循环错误: {e}")
            time.sleep(300)

# ✅ 启动交易机器人
trading_bot()