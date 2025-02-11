import os
import ccxt
import gym
import numpy as np
import pandas as pd
import time
import logging
import pickle
from stable_baselines3 import PPO
from datetime import datetime
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ✅ 统一日志文件
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# ✅ OKX API 配置
exchange = ccxt.okx({
    'apiKey': "0f046e6a-1627-4db4-b97d-083d7e6cc16b",
    'secret': "BF7BC880C73AD54D2528FA271A358C2C",
    'password': "Duan0918.",
    'options': {'defaultType': 'swap'},
})

# ✅ 交易参数（自动调整）
symbols = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP", "ADA-USDT-SWAP"]
update_interval = 1800  # 每 30 分钟更新模型
risk_percentage = 0.1  # 每笔交易使用余额的 10%

# ✅ 获取账户余额（自动计算交易量）
def get_trade_amount(symbol):
    try:
        balance = exchange.fetch_balance()
        usdt_balance = balance['total']['USDT']
        ticker = exchange.fetch_ticker(symbol)
        price = ticker['last']

        trade_amount = (usdt_balance * risk_percentage) / price
        return round(trade_amount, 4)
    except Exception as e:
        logging.error(f"❌ 获取交易金额失败: {symbol}，错误: {e}")
        return 0.01  # 失败时使用默认值

# ✅ 获取市场数据（计算波动率 & 交易参数）
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
        df['volatility'] = df['close'].pct_change().rolling(20).std()
        df = df.dropna()

        return df
    except Exception as e:
        logging.error(f"❌ 获取市场数据失败: {symbol}，错误: {e}")
        return None

# ✅ 训练或更新随机森林模型
def train_rf_model():
    all_data = []
    for symbol in symbols:
        data = get_market_data(symbol)
        if data is not None:
            data['label'] = np.where(data['close'].diff() > 0, 1, 0)
            all_data.append(data)

    if all_data:
        new_data = pd.concat(all_data, ignore_index=True)

        if os.path.exists("rf_training_data.pkl"):
            with open("rf_training_data.pkl", "rb") as f:
                old_data = pickle.load(f)
            full_data = pd.concat([old_data, new_data], ignore_index=True)
        else:
            full_data = new_data

        X = full_data[['ma5', 'ma15', 'atr', 'rsi', 'macd', 'bollinger_up', 'bollinger_down']]
        y = full_data['label']

        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, warm_start=True)
        rf_model.fit(X, y)

        with open("random_forest_model.pkl", "wb") as f:
            pickle.dump(rf_model, f)
        with open("rf_training_data.pkl", "wb") as f:
            pickle.dump(full_data, f)

        logging.info("✅ 随机森林模型训练完成")

# ✅ 获取交易信号（结合 PPO & 随机森林）
def get_trade_signal(symbol):
    try:
        model = PPO.load(f"ppo_trading_agent_{symbol}.zip")
    except:
        logging.warning(f"⚠️ 未找到 PPO 模型 {symbol}，跳过交易")
        return "hold"

    data = get_market_data(symbol)
    if data is None:
        return "hold"

    state = data.iloc[-1][['ma5', 'ma15', 'atr', 'rsi', 'macd', 'bollinger_up', 'bollinger_down']].values
    action, _ = model.predict(state)
    
    if action == 1:
        execute_trade(symbol, "buy")
    else:
        execute_trade(symbol, "sell")

# ✅ 执行交易（自动调整交易量）
def execute_trade(symbol, action):
    trade_amount = get_trade_amount(symbol)
    if trade_amount <= 0:
        logging.warning(f"⚠️ {symbol} 交易量过低，跳过交易")
        return

    try:
        if action == "buy":
            exchange.create_market_order(symbol, "buy", trade_amount)
            logging.info(f"✅ 买入 {symbol}，数量: {trade_amount}")
        elif action == "sell":
            exchange.create_market_order(symbol, "sell", trade_amount)
            logging.info(f"✅ 卖出 {symbol}，数量: {trade_amount}")
    except Exception as e:
        logging.error(f"❌ 交易失败: {symbol}，错误: {e}")

# ✅ 根据市场波动率自动调整交易频率
def get_dynamic_sleep_time(symbol):
    data = get_market_data(symbol)
    if data is None:
        return 300

    volatility = data['volatility'].iloc[-1]
    if volatility > 0.02:
        return 120
    elif volatility > 0.01:
        return 300
    else:
        return 600

# ✅ 交易循环（自动调整交易频率）
if __name__ == "__main__":
    train_rf_model()
    last_update_time = time.time()

    while True:
        for symbol in symbols:
            get_trade_signal(symbol)

        if time.time() - last_update_time > update_interval:
            train_rf_model()
            last_update_time = time.time()

        sleep_time = min(get_dynamic_sleep_time(symbol) for symbol in symbols)
        logging.info(f"⏳ 休眠 {sleep_time} 秒后继续交易")
        time.sleep(sleep_time)