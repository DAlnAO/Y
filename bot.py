import os
import ccxt
import pandas as pd
import numpy as np
import time
import logging
from stable_baselines3 import SAC
from sklearn.preprocessing import MinMaxScaler

# ✅ 设置日志
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# ✅ OKX API 配置
exchange = ccxt.okx({
    'apiKey': "0f046e6a-1627-4db4-b97d-083d7e6cc16b",
    'secret': "BF7BC880C73AD54D2528FA271A358C2C",
    'password': "Duan0918.",
    'options': {'defaultType': 'swap'},
})

# ✅ 交易参数
risk_percentage = 10  # 只使用 10% 账户余额交易
max_drawdown = 15
min_leverage = 5
max_leverage = 125
trade_history_file = "trade_history.csv"
symbols = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"]
model_path = "trading_model.zip"
training_interval = 86400  # **每 24 小时重新训练**
cooldown_base = 120  # **基础冷却时间 120 秒**
last_trade_time = {symbol: 0 for symbol in symbols}  # **记录上次交易时间**

# ✅ 确保交易数据文件存在
if not os.path.exists(trade_history_file):
    pd.DataFrame(columns=["timestamp", "symbol", "action", "size", "price", "pnl"]).to_csv(trade_history_file, index=False)

# ✅ 获取市场数据
def get_market_data(symbol, timeframe='5m', limit=500):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma15'] = df['close'].rolling(window=15).mean()
        df['rsi'] = 100 - (100 / (1 + df['close'].pct_change().rolling(14).mean()))
        df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        df['atr'] = df['high'].rolling(14).max() - df['low'].rolling(14).min()
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()

        df = df.dropna()
        return df
    except Exception as e:
        logging.error(f"⚠️ 获取市场数据失败: {e}")
        return None

# ✅ 训练强化学习模型
def train_rl_model():
    df = pd.read_csv(trade_history_file)
    
    if len(df) < 500:
        logging.warning(f"⚠️ 训练数据不足 ({len(df)}/500)，强化学习跳过，使用默认策略")
        return "default"

    logging.info("🔄 开始训练强化学习模型...")
    start_time = time.time()
    
    env_data = df[['price', 'pnl']].values
    scaler = MinMaxScaler()
    env_data = scaler.fit_transform(env_data)

    model = SAC("MlpPolicy", env_data, verbose=1)
    model.learn(total_timesteps=20000)
    model.save(model_path)

    training_duration = time.time() - start_time
    logging.info(f"✅ 强化学习模型训练完成，训练时间: {training_duration:.2f} 秒，已更新！")
    return model

# ✅ 计算智能杠杆
def get_dynamic_leverage(symbol):
    df = get_market_data(symbol)
    if df is None or len(df) < 20:
        return min_leverage

    atr = df['atr'].rolling(20).mean().iloc[-1]
    volatility = atr / df['close'].iloc[-1]
    leverage = int(np.clip((30 - volatility * 3000), min_leverage, max_leverage))

    logging.info(f"🔄 智能杠杆: {symbol} | 波动率: {volatility:.4f} | 设定杠杆: {leverage}x")
    return leverage

# ✅ 计算动态冷却时间
def get_dynamic_cooldown(symbol):
    df = get_market_data(symbol)
    if df is None or len(df) < 20:
        return cooldown_base

    atr = df['atr'].rolling(20).mean().iloc[-1]
    volatility = atr / df['close'].iloc[-1]

    cooldown = int(np.clip((cooldown_base + volatility * 5000), cooldown_base, 1800))  # **最大冷却 30 分钟**
    logging.info(f"⏳ 交易冷却时间: {symbol} | 波动率: {volatility:.4f} | 设定冷却: {cooldown} 秒")
    return cooldown

# ✅ 获取交易信号
def get_trade_signal(symbol, model):
    df = get_market_data(symbol)
    if df is None or len(df) < 10:
        return "hold"

    if model == "default":
        strategy = "均线策略"
        if df['ma5'].iloc[-1] > df['ma15'].iloc[-1]:
            return "buy", strategy
        elif df['ma5'].iloc[-1] < df['ma15'].iloc[-1]:
            return "sell", strategy
        return "hold", strategy

    strategy = "强化学习"
    action, _states = model.predict(df[['ma5', 'ma15', 'rsi', 'macd', 'atr', 'obv']].values[-10:].reshape(1, 10, 6))
    
    if action == 0:
        return "buy", strategy
    elif action == 1:
        return "sell", strategy
    return "hold", strategy

# ✅ 交易机器人
def trading_bot():
    last_training_time = time.time()

    # **初始化模型**
    if os.path.exists(model_path):
        model = SAC.load(model_path)
        logging.info("✅ 加载已有强化学习模型")
    else:
        model = train_rl_model()

    while True:
        try:
            balance = exchange.fetch_balance()
            usdt_balance = balance['total'].get('USDT', 0)

            for symbol in symbols:
                current_time = time.time()
                cooldown = get_dynamic_cooldown(symbol)

                if current_time - last_trade_time[symbol] < cooldown:
                    logging.info(f"🚫 冷却中: {symbol} | 等待 {cooldown - (current_time - last_trade_time[symbol]):.2f} 秒")
                    continue

                leverage = get_dynamic_leverage(symbol)
                signal, strategy = get_trade_signal(symbol, model)

                logging.info(f"📢 交易信号: {symbol} | {signal.upper()} | 由 {strategy} 生成")

                if signal in ["buy", "sell"]:
                    trade_size = round((usdt_balance * (risk_percentage / 100)), 2)
                    exchange.set_leverage(leverage, symbol, params={"mgnMode": "isolated"})

                    try:
                        order = exchange.create_market_order(symbol, signal, trade_size)
                        last_trade_time[symbol] = current_time
                        logging.info(f"✅ 交易成功: {signal.upper()} {trade_size} 张 {symbol} | 杠杆: {leverage}x")
                    except Exception as e:
                        logging.error(f"⚠️ 交易失败: {e}")

            logging.info(f"💰 账户余额: {usdt_balance} USDT")
            time.sleep(60)  # **每分钟检查市场**
        
        except Exception as e:
            logging.error(f"⚠️ 交易循环错误: {e}")
            time.sleep(60)

# ✅ 启动机器人
trading_bot()