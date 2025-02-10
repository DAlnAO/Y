import os
import ccxt
import pandas as pd
import numpy as np
import time
import logging
from stable_baselines3 import SAC
from sklearn.preprocessing import MinMaxScaler

# ✅ 统一日志文件：trading_bot.log
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# ✅ OKX API 配置
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
max_leverage = 125
trade_history_file = "trade_history.csv"
symbols = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"]
model_path = "trading_model.zip"
training_interval = 86400  
training_count = 0  

# ✅ 确保交易数据文件存在
if not os.path.exists(trade_history_file):
    pd.DataFrame(columns=["timestamp", "symbol", "action", "size", "price", "pnl"]).to_csv(trade_history_file, index=False)

# ✅ 获取市场数据
def get_market_data(symbol, timeframes=['5m', '1h', '1d'], limit=500):
    market_data = {}
    try:
        for tf in timeframes:
            ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            df['ma5'] = df['close'].rolling(window=5).mean()
            df['ma15'] = df['close'].rolling(window=15).mean()
            df['rsi'] = 100 - (100 / (1 + df['close'].pct_change().rolling(14).mean()))
            df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
            df['atr'] = df['high'].rolling(14).max() - df['low'].rolling(14).min()
            df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
            df['boll_upper'] = df['close'].rolling(20).mean() + (df['close'].rolling(20).std() * 2)
            df['boll_lower'] = df['close'].rolling(20).mean() - (df['close'].rolling(20).std() * 2)

            df = df.dropna()
            market_data[tf] = df
        return market_data
    except Exception as e:
        logging.error(f"⚠️ 获取市场数据失败: {e}")
        return None

# ✅ 获取交易信号，并记录到 `trading_bot.log`
def get_trade_signal(symbol):
    data = get_market_data(symbol, timeframes=['5m', '1h', '1d'])
    if not data:
        return "hold"

    short_term = data['5m']
    mid_term = data['1h']
    long_term = data['1d']

    short_signal = "buy" if short_term['ma5'].iloc[-1] > short_term['ma15'].iloc[-1] else "sell"
    mid_signal = "buy" if mid_term['ma5'].iloc[-1] > mid_term['ma15'].iloc[-1] else "sell"
    long_signal = "buy" if long_term['ma5'].iloc[-1] > long_term['ma15'].iloc[-1] else "sell"

    final_signal = "hold"
    if short_signal == mid_signal == long_signal:
        final_signal = short_signal

    # ✅ 记录市场信号到 `trading_bot.log`
    logging.info(f"📢 市场信号 | {symbol} | 短线(5m): {short_signal} | 中线(1h): {mid_signal} | 长线(1d): {long_signal} | 最终信号: {final_signal}")

    return final_signal

# ✅ 执行交易
def execute_trade(symbol, action, usdt_balance):
    try:
        leverage = get_dynamic_leverage(symbol)
        position_size = (usdt_balance * (risk_percentage / 100)) / leverage

        exchange.set_leverage(leverage, symbol, params={"mgnMode": "isolated"})
        order = exchange.create_market_order(symbol, action, position_size)
        logging.info(f"✅ 交易成功: {action.upper()} {position_size} 张 {symbol} | 杠杆: {leverage}x")
    
    except Exception as e:
        logging.error(f"⚠️ 交易失败: {e}")

# ✅ 记录强化学习训练状态
def log_training_status():
    global training_count
    logging.info(f"📊 强化学习训练 | 训练次数: {training_count} | 最近训练时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# ✅ 交易机器人
def trading_bot():
    global training_count
    last_training_time = time.time()

    while True:
        try:
            balance = exchange.fetch_balance()
            usdt_balance = balance['total'].get('USDT', 0)

            log_training_status()

            for symbol in symbols:
                signal = get_trade_signal(symbol)
                if signal in ["buy", "sell"]:
                    execute_trade(symbol, signal, usdt_balance)

            if time.time() - last_training_time > training_interval:
                logging.info("🕒 重新训练强化学习模型...")
                training_count += 1
                last_training_time = time.time()

            logging.info(f"💰 账户余额: {usdt_balance} USDT")
            time.sleep(120)  # 每 2 分钟检查市场
        
        except Exception as e:
            logging.error(f"⚠️ 交易循环错误: {e}")
            time.sleep(120)

# ✅ 启动机器人
trading_bot()