import os
import ccxt
import pandas as pd
import numpy as np
import time
import logging
from sklearn.preprocessing import MinMaxScaler

# ✅ 设置日志系统
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# ✅ OKX API 配置（已直接写入你的 API 信息）
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
        df['rsi'] = 100 - (100 / (1 + df['close'].pct_change().rolling(14).mean()))
        df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        df['atr'] = df['high'].rolling(14).max() - df['low'].rolling(14).min()
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()

        df = df.dropna()
        return df
    except Exception as e:
        logging.error(f"⚠️ 获取市场数据失败: {e}")
        return None

# ✅ 计算智能杠杆
def get_dynamic_leverage(symbol):
    df = get_market_data(symbol)
    if df is None or len(df) < 20:
        return min_leverage

    atr = df['atr'].rolling(20).mean().iloc[-1]
    volatility = atr / df['close'].iloc[-1]
    leverage = np.clip(int(50 - volatility * 5000), min_leverage, max_leverage)

    logging.info(f"🔄 智能杠杆: {symbol} | 波动率: {volatility:.4f} | 杠杆: {leverage}x")
    return leverage

# ✅ 获取交易信号（基于技术分析）
def get_trade_signal(symbol):
    df = get_market_data(symbol)
    if df is None or len(df) < 10:
        return "hold", 0, 0

    ma5 = df['ma5'].iloc[-1]
    ma15 = df['ma15'].iloc[-1]
    rsi = df['rsi'].iloc[-1]
    macd = df['macd'].iloc[-1]
    atr = df['atr'].iloc[-1]
    close_price = df['close'].iloc[-1]

    # 确定买卖信号
    if ma5 > ma15 and rsi > 50 and macd > 0:
        return "buy", close_price - atr * 1.5, close_price + atr * 2
    elif ma5 < ma15 and rsi < 50 and macd < 0:
        return "sell", close_price + atr * 1.5, close_price - atr * 2
    else:
        return "hold", 0, 0

# ✅ 执行交易，并记录数据
def execute_trade(symbol, action, size, stop_loss, take_profit, leverage):
    try:
        order = exchange.create_market_order(symbol, action, size)
        price = exchange.fetch_ticker(symbol)['last']
        timestamp = pd.Timestamp.now()

        # 记录交易数据
        trade_data = {
            "timestamp": timestamp,
            "symbol": symbol,
            "action": action,
            "size": size,
            "price": price,
            "pnl": 0  # 初始 PnL 设为 0，后续更新
        }

        df = pd.DataFrame([trade_data])
        df.to_csv(trade_history_file, mode='a', header=False, index=False)

        logging.info(f"✅ 交易成功: {action.upper()} {size} 张 {symbol} - 价格: {price}, 杠杆: {leverage}x")
    except Exception as e:
        logging.error(f"⚠️ 交易失败: {e}")

# ✅ 交易机器人
def trading_bot():
    initial_balance = exchange.fetch_balance()['total'].get('USDT', 0)

    while True:
        try:
            balance = exchange.fetch_balance()
            usdt_balance = balance['total'].get('USDT', 0)

            for symbol in symbols:
                leverage = get_dynamic_leverage(symbol)
                signal, stop_loss, take_profit = get_trade_signal(symbol)

                if signal in ["buy", "sell"]:
                    trade_size = round((usdt_balance * (risk_percentage / 100)), 2)
                    execute_trade(symbol, signal, trade_size, stop_loss, take_profit, leverage)

            # ✅ 交易数据积累到 500 条后，触发强化学习训练
            df = pd.read_csv(trade_history_file)
            if len(df) >= 500:
                logging.info("🕒 交易数据已累积 500 条，触发强化学习训练...")
                train_rl_model()

            # ✅ 记录账户余额
            logging.info(f"💰 账户余额: {usdt_balance} USDT")
            time.sleep(300)

        except Exception as e:
            logging.error(f"⚠️ 交易循环错误: {e}")
            time.sleep(300)

# ✅ 启动交易机器人
trading_bot()