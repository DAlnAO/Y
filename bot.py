import os
import ccxt
import pandas as pd
import numpy as np
import time
import logging
from stable_baselines3 import SAC
from sklearn.preprocessing import MinMaxScaler

# ✅ 日志系统
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# ✅ OKX API 配置
exchange = ccxt.okx({
    'apiKey': "0f046e6a-1627-4db4-b97d-083d7e6cc16b",
    'secret': "BF7BC880C73AD54D2528FA271A358C2C",
    'password': "Duan0918.",
    'options': {'defaultType': 'swap'},
})

# ✅ 交易参数
risk_percentage = 5  # 降低交易风险，每次交易 5% 账户资金
max_drawdown = 15
min_leverage = 5
max_leverage = 50
trade_history_file = "trade_history.csv"
symbols = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"]
model_path = "trading_model.zip"

# ✅ 确保交易数据文件存在
if not os.path.exists(trade_history_file):
    pd.DataFrame(columns=["timestamp", "symbol", "action", "size", "price", "pnl"]).to_csv(trade_history_file, index=False)

# ✅ 获取市场数据
def get_market_data(symbol, timeframe='5m', limit=500):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # 计算技术指标
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma15'] = df['close'].rolling(window=15).mean()
        df['atr'] = df['high'].rolling(14).max() - df['low'].rolling(14).min()
        df['volatility'] = df['atr'] / df['close']

        df = df.dropna()
        return df
    except Exception as e:
        logging.error(f"⚠️ 获取市场数据失败: {e}")
        return None

# ✅ 计算智能杠杆（避免爆仓）
def get_dynamic_leverage(symbol):
    df = get_market_data(symbol)
    if df is None or len(df) < 20:
        return min_leverage

    volatility = df['volatility'].rolling(20).mean().iloc[-1]
    leverage = np.clip(int(50 - volatility * 5000), min_leverage, max_leverage)

    logging.info(f"🔄 智能杠杆: {symbol} | 波动率: {volatility:.4f} | 设定杠杆: {leverage}x")
    return leverage

# ✅ 训练强化学习模型
def train_rl_model():
    df = pd.read_csv(trade_history_file)
    
    if len(df) < 500:
        logging.warning(f"⚠️ 训练数据不足 ({len(df)}/500)，强化学习跳过，使用均线策略")
        return "default"

    env_data = df[['price', 'pnl']].values
    scaler = MinMaxScaler()
    env_data = scaler.fit_transform(env_data)

    model = SAC("MlpPolicy", env_data, verbose=1)
    model.learn(total_timesteps=20000)
    model.save(model_path)

    return model

# ✅ 获取交易信号（支持强化学习 & 均线策略）
def get_trade_signal(symbol, model):
    df = get_market_data(symbol)
    if df is None or len(df) < 10:
        return "hold", 0, 0

    atr = df['atr'].iloc[-1]

    if model == "default":
        if df['ma5'].iloc[-1] > df['ma15'].iloc[-1]:
            return "buy", df['close'].iloc[-1] - atr * 1.5, df['close'].iloc[-1] + atr * 2
        else:
            return "sell", df['close'].iloc[-1] - atr * 1.5, df['close'].iloc[-1] + atr * 2

    features = df[['ma5', 'ma15', 'atr']].values[-10:]
    action, _states = model.predict(features.reshape(1, 10, 3))

    if action == 0:
        return "buy", df['close'].iloc[-1] - atr * 1.5, df['close'].iloc[-1] + atr * 2
    elif action == 1:
        return "sell", df['close'].iloc[-1] - atr * 1.5, df['close'].iloc[-1] + atr * 2
    else:
        return "hold", 0, 0

# ✅ 执行交易（支持资金检查 & 逐仓模式）
def execute_trade(symbol, action, size, stop_loss, take_profit, leverage):
    try:
        balance = exchange.fetch_balance()
        usdt_balance = balance['total'].get('USDT', 0)

        if usdt_balance < size * 1.1:
            logging.error(f"⚠️ 资金不足，跳过交易: {symbol} 需要 {size}, 账户余额 {usdt_balance}")
            return

        exchange.set_margin_mode("isolated", symbol)
        exchange.set_leverage(leverage, symbol)

        order = exchange.create_market_order(symbol, action, size)
        logging.info(f"✅ 交易成功: {action.upper()} {size} 张 {symbol} - 止损: {stop_loss}, 止盈: {take_profit}, 杠杆: {leverage}x")
    except Exception as e:
        logging.error(f"⚠️ 交易失败: {e}")

# ✅ 交易机器人（智能调整交易频率，避免频繁交易）
def trading_bot():
    initial_balance = exchange.fetch_balance()['total'].get('USDT', 0)
    
    if os.path.exists(model_path):
        model = SAC.load(model_path)
    else:
        model = train_rl_model()

    while True:
        try:
            balance = exchange.fetch_balance()
            usdt_balance = balance['total'].get('USDT', 0)

            for symbol in symbols:
                leverage = get_dynamic_leverage(symbol)
                signal, stop_loss, take_profit = get_trade_signal(symbol, model)
                if signal in ["buy", "sell"]:
                    trade_size = round((usdt_balance * (risk_percentage / 100)), 2)
                    execute_trade(symbol, signal, trade_size, stop_loss, take_profit, leverage)

            if ((usdt_balance - initial_balance) / initial_balance) * 100 <= -max_drawdown:
                break

            logging.info(f"💰 账户余额: {usdt_balance} USDT")
            time.sleep(120)  # 2分钟调整交易频率

        except Exception as e:
            logging.error(f"⚠️ 交易循环错误: {e}")
            time.sleep(120)

# ✅ 启动机器人
trading_bot()