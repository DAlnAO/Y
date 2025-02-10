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
risk_percentage = 10  # **每次交易使用账户余额的 10%**
max_drawdown = 15
min_leverage = 5
max_leverage = 125  # **修正：OKX 最大支持 125 倍杠杆**
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

# ✅ 计算智能杠杆（确保 1~125 倍杠杆）
def get_dynamic_leverage(symbol):
    df = get_market_data(symbol)
    if df is None or len(df) < 20:
        return min_leverage

    atr = df['atr'].rolling(20).mean().iloc[-1]
    volatility = atr / df['close'].iloc[-1]
    leverage = int(np.clip((30 - volatility * 3000), min_leverage, max_leverage))

    logging.info(f"🔄 智能杠杆: {symbol} | 波动率: {volatility:.4f} | 设定杠杆: {leverage}x")
    return leverage

# ✅ 计算持仓大小（**使用账户余额的 10%**）
def calculate_position_size(symbol, usdt_balance, leverage):
    price = get_market_data(symbol)['close'].iloc[-1]
    risk_allocation = usdt_balance * (risk_percentage / 100)  # **仅使用 10% 资金**
    position_size = (risk_allocation * leverage) / price

    logging.info(f"💰 计算持仓: {symbol} | 账户余额: {usdt_balance} USDT | 使用资金: {risk_allocation} USDT | 杠杆: {leverage}x | 交易张数: {round(position_size, 3)}")
    return round(position_size, 3)

# ✅ 获取交易信号
def get_trade_signal(symbol):
    df = get_market_data(symbol)
    if df is None or len(df) < 10:
        return "hold"

    if df['ma5'].iloc[-1] > df['ma15'].iloc[-1]:
        return "buy"
    elif df['ma5'].iloc[-1] < df['ma15'].iloc[-1]:
        return "sell"
    return "hold"

# ✅ 执行交易（**使用 10% 账户余额**）
def execute_trade(symbol, action, usdt_balance):
    try:
        leverage = get_dynamic_leverage(symbol)
        position_size = calculate_position_size(symbol, usdt_balance, leverage)

        # ✅ 设置逐仓模式
        exchange.set_leverage(leverage, symbol, params={"mgnMode": "isolated"})

        balance = exchange.fetch_balance()
        available_margin = balance['free'].get('USDT', 0)
        if available_margin < position_size * leverage:
            logging.warning(f"⚠️ 交易失败: 账户保证金不足 | 可用: {available_margin} USDT | 需要: {position_size * leverage} USDT")
            return

        order = exchange.create_market_order(symbol, action, position_size)
        logging.info(f"✅ 交易成功: {action.upper()} {position_size} 张 {symbol} | 杠杆: {leverage}x")
    
    except Exception as e:
        logging.error(f"⚠️ 交易失败: {e}")

# ✅ 交易机器人（每 2 分钟检查一次市场）
def trading_bot():
    initial_balance = exchange.fetch_balance()['total'].get('USDT', 0)
    
    while True:
        try:
            balance = exchange.fetch_balance()
            usdt_balance = balance['total'].get('USDT', 0)

            for symbol in symbols:
                leverage = get_dynamic_leverage(symbol)
                signal = get_trade_signal(symbol)

                if signal in ["buy", "sell"]:
                    execute_trade(symbol, signal, usdt_balance)

            if ((usdt_balance - initial_balance) / initial_balance) * 100 <= -max_drawdown:
                logging.warning("⚠️ 账户亏损超过最大限制，停止交易！")
                break

            logging.info(f"💰 账户余额: {usdt_balance} USDT")
            time.sleep(120)  # **每 2 分钟检查一次市场**
        
        except Exception as e:
            logging.error(f"⚠️ 交易循环错误: {e}")
            time.sleep(120)

# ✅ 启动机器人
trading_bot()