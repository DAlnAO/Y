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

# ✅ OKX API 配置（使用逐仓模式）
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
interval_seconds = 120  # 每 2 分钟检查一次市场

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

# ✅ 强制设置逐仓模式
def set_isolated_margin_mode(symbol, leverage):
    try:
        params = {
            "instId": symbol,
            "lever": str(leverage),
            "mgnMode": "isolated"
        }
        exchange.private_post_account_set_leverage(params)
        logging.info(f"✅ 已设置 {symbol} 为逐仓模式，杠杆: {leverage}x")
    except Exception as e:
        logging.error(f"⚠️ 设置杠杆失败: {e}")

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

# ✅ 获取交易信号
def get_trade_signal(symbol):
    df = get_market_data(symbol)
    if df is None or len(df) < 10:
        return "hold", 0, 0

    atr = df['atr'].iloc[-1]

    # ✅ 简单均线策略（替代强化学习）
    if df['ma5'].iloc[-1] > df['ma15'].iloc[-1]:
        return "buy", df['close'].iloc[-1] - atr * 1.5, df['close'].iloc[-1] + atr * 2
    else:
        return "sell", df['close'].iloc[-1] - atr * 1.5, df['close'].iloc[-1] + atr * 2

# ✅ 执行交易
def execute_trade(symbol, action, size, stop_loss, take_profit, leverage):
    try:
        set_isolated_margin_mode(symbol, leverage)
        order = exchange.create_market_order(symbol, action, size)
        logging.info(f"✅ 交易成功: {action.upper()} {size} 张 {symbol} - 止损: {stop_loss}, 止盈: {take_profit}, 杠杆: {leverage}x")
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

                logging.info(f"📢 {symbol} 交易信号: {signal} | 当前价格: {exchange.fetch_ticker(symbol)['last']} USDT")

                if signal in ["buy", "sell"]:
                    trade_size = round((usdt_balance * (risk_percentage / 100)), 2)
                    execute_trade(symbol, signal, trade_size, stop_loss, take_profit, leverage)

            # ✅ 每 2 分钟检查市场并反馈日志
            logging.info(f"💰 账户余额: {usdt_balance} USDT，等待 {interval_seconds / 60} 分钟后继续...")
            time.sleep(interval_seconds)

        except Exception as e:
            logging.error(f"⚠️ 交易循环错误: {e}")
            time.sleep(interval_seconds)

# ✅ 启动机器人
trading_bot()
