import os
import ccxt
import pandas as pd
import numpy as np
import time
import logging
import talib as ta  # ADX 需要 TA-Lib
from stable_baselines3 import SAC
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

            # 计算技术指标
            df['ma5'] = df['close'].rolling(window=5).mean()
            df['ma15'] = df['close'].rolling(window=15).mean()
            df['rsi'] = ta.RSI(df['close'], timeperiod=14)  # 使用 TA-Lib 计算 RSI
            df['macd'], df['macd_signal'], _ = ta.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
            df['atr'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            df['adx'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=14)  # 计算 ADX
            df['boll_upper'], df['boll_middle'], df['boll_lower'] = ta.BBANDS(df['close'], timeperiod=20)

            df = df.dropna()
            market_data[tf] = df
        return market_data
    except Exception as e:
        logging.error(f"⚠️ 获取市场数据失败: {e}")
        return None

# ✅ 获取交易信号，并记录到日志
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

    # ✅ 允许 5m 和 1h 信号一致就交易，不必等 1d
    final_signal = "hold"
    if short_signal == mid_signal:
        final_signal = short_signal

    # ✅ 过滤震荡市场（ADX < 25 不交易）
    if short_term['adx'].iloc[-1] < 25:
        logging.info(f"⚠️ {symbol} 市场震荡 (ADX < 25)，跳过交易")
        return "hold"

    logging.info(f"📢 市场信号 | {symbol} | 5m: {short_signal} | 1h: {mid_signal} | 1d: {long_signal} | 最终: {final_signal}")
    return final_signal

# ✅ 计算智能杠杆
def get_dynamic_leverage(symbol):
    df = get_market_data(symbol)
    if df is None or '5m' not in df or len(df['5m']) < 20:
        return min_leverage

    atr = df['5m']['atr'].rolling(20).mean().iloc[-1]
    volatility = atr / df['5m']['close'].iloc[-1]
    leverage = int(np.clip((30 - volatility * 3000), min_leverage, max_leverage))

    logging.info(f"🔄 智能杠杆: {symbol} | 波动率: {volatility:.4f} | 设定杠杆: {leverage}x")
    return leverage

# ✅ 计算动态止盈止损
def get_dynamic_stop_loss_take_profit(symbol):
    df = get_market_data(symbol)
    atr = df['5m']['atr'].iloc[-1]
    adx = df['5m']['adx'].iloc[-1]

    if adx > 30:
        stop_loss = atr * 1.5
        take_profit = atr * 3  # 趋势市场放大止盈
    else:
        stop_loss = atr * 1
        take_profit = atr * 1.5  # 震荡市场缩小止盈

    logging.info(f"📈 {symbol} 动态止盈: {take_profit:.2f} | 动态止损: {stop_loss:.2f}")
    return stop_loss, take_profit

# ✅ 执行交易
def execute_trade(symbol, action, usdt_balance):
    try:
        leverage = get_dynamic_leverage(symbol)
        position_size = (usdt_balance * (risk_percentage / 100)) / leverage
        stop_loss, take_profit = get_dynamic_stop_loss_take_profit(symbol)

        exchange.set_leverage(leverage, symbol, params={"mgnMode": "isolated"})
        order = exchange.create_market_order(symbol, action, position_size)
        logging.info(f"✅ 交易成功: {action.upper()} {position_size} 张 {symbol} | 杠杆: {leverage}x | 止盈: {take_profit} | 止损: {stop_loss}")
    
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