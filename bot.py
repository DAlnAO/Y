import os
import ccxt
import pandas as pd
import numpy as np
import time
import logging
from stable_baselines3 import SAC
from sklearn.preprocessing import MinMaxScaler

# ✅ 设置日志系统
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# ✅ OKX API 配置（逐仓模式）
exchange = ccxt.okx({
    'apiKey': "0f046e6a-1627-4db4-b97d-083d7e6cc16b",
    'secret': "BF7BC880C73AD54D2528FA271A358C2C",
    'password': "Duan0918.",
    'options': {'defaultType': 'swap'},
})

# ✅ 交易参数
risk_percentage = 10
max_drawdown = 15
min_leverage = 5
max_leverage = 20  # 最高 20x，避免超出保证金
trade_history_file = "trade_history.csv"
symbols = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"]
model_path = "trading_model.zip"

# ✅ 确保交易数据文件存在
if not os.path.exists(trade_history_file):
    pd.DataFrame(columns=["timestamp", "symbol", "action", "size", "price", "pnl"]).to_csv(trade_history_file, index=False)

# ✅ 获取账户余额
def check_balance():
    try:
        balance = exchange.fetch_balance()
        usdt_balance = balance['total'].get('USDT', 0)
        logging.info(f"💰 当前账户 USDT 余额: {usdt_balance}")
        return usdt_balance
    except Exception as e:
        logging.error(f"⚠️ 获取账户余额失败: {e}")
        return 0

# ✅ 获取市场数据
def get_market_data(symbol, timeframe='5m', limit=500):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # 计算技术指标
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

# ✅ 训练强化学习模型（如果数据不足，则使用默认均线策略）
def train_rl_model():
    df = pd.read_csv(trade_history_file)
    
    if len(df) < 500:
        logging.warning(f"⚠️ 训练数据不足 ({len(df)}/500)，强化学习跳过，使用默认策略")
        return "default"

    env_data = df[['price', 'pnl']].values
    scaler = MinMaxScaler()
    env_data = scaler.fit_transform(env_data)

    model = SAC("MlpPolicy", env_data, verbose=1)
    model.learn(total_timesteps=20000)
    model.save(model_path)

    return model

# ✅ 计算智能杠杆（限制最大 20x）
def get_dynamic_leverage(symbol):
    df = get_market_data(symbol)
    if df is None or len(df) < 20:
        return min_leverage

    atr = df['atr'].rolling(20).mean().iloc[-1]
    volatility = atr / df['close'].iloc[-1]
    leverage = np.clip(int(30 - volatility * 3000), min_leverage, max_leverage)

    logging.info(f"🔄 智能杠杆: {symbol} | 波动率: {volatility:.4f} | 杠杆: {leverage}x")
    return leverage

# ✅ 计算交易金额（最多使用账户余额的 10%）
def get_trade_size(usdt_balance, leverage):
    min_trade = 10  # 最小交易金额 10 USDT
    max_trade = (usdt_balance * 0.1) / leverage  # 调整杠杆后的交易金额
    trade_size = round(max_trade, 2)

    if trade_size < min_trade:
        logging.warning(f"⚠️ 账户余额太低，最小交易金额 {min_trade} USDT")
        return 0  # 余额过低，不进行交易

    return trade_size

# ✅ 获取交易信号（如果模型不可用，则使用均线策略）
def get_trade_signal(symbol, model):
    df = get_market_data(symbol)
    if df is None or len(df) < 10:
        return "hold", 0, 0

    features = df[['ma5', 'ma15', 'rsi', 'macd', 'atr', 'obv']].values[-10:]
    atr = df['atr'].iloc[-1]

    if model == "default":
        if df['ma5'].iloc[-1] > df['ma15'].iloc[-1]:
            return "buy", df['close'].iloc[-1] - atr * 1.5, df['close'].iloc[-1] + atr * 2
        else:
            return "sell", df['close'].iloc[-1] - atr * 1.5, df['close'].iloc[-1] + atr * 2

    action, _states = model.predict(features.reshape(1, 10, 6))
    return ("buy", df['close'].iloc[-1] - atr * 1.5, df['close'].iloc[-1] + atr * 2) if action == 0 else ("sell", df['close'].iloc[-1] - atr * 1.5, df['close'].iloc[-1] + atr * 2)

# ✅ 执行交易（逐仓模式）
def execute_trade(symbol, action, size, stop_loss, take_profit, leverage):
    try:
        exchange.set_margin_mode('isolated', symbol)  # 设置逐仓模式
        order = exchange.create_market_order(symbol, action, size)
        logging.info(f"✅ 交易成功: {action.upper()} {size} 张 {symbol} - 止损: {stop_loss}, 止盈: {take_profit}, 杠杆: {leverage}x")
    except Exception as e:
        logging.error(f"⚠️ 交易失败: {e}")

# ✅ 交易机器人（每 2 分钟运行一次）
def trading_bot():
    initial_balance = check_balance()
    
    if os.path.exists(model_path):
        model = SAC.load(model_path)
    else:
        model = train_rl_model()

    while True:
        try:
            usdt_balance = check_balance()

            for symbol in symbols:
                leverage = get_dynamic_leverage(symbol)
                trade_size = get_trade_size(usdt_balance, leverage)
                signal, stop_loss, take_profit = get_trade_signal(symbol, model)

                if trade_size > 0 and signal in ["buy", "sell"]:
                    execute_trade(symbol, signal, trade_size, stop_loss, take_profit, leverage)

            if ((usdt_balance - initial_balance) / initial_balance) * 100 <= -max_drawdown:
                break

            time.sleep(120)  # 每 2 分钟检查市场
        except Exception as e:
            logging.error(f"⚠️ 交易循环错误: {e}")
            time.sleep(120)

# ✅ 启动机器人
trading_bot()