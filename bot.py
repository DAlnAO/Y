import os
import ccxt
import pandas as pd
import numpy as np
import time
import logging
from stable_baselines3 import SAC

# ✅ 设置日志
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# ✅ OKX API 配置
exchange = ccxt.okx({
    'apiKey': "你的API_KEY",
    'secret': "你的SECRET",
    'password': "你的密码",
    'options': {'defaultType': 'swap'},
})

# ✅ 交易参数
risk_percentage = 10  
max_drawdown = 15
min_leverage = 5
max_leverage = 125
trade_history_file = "trade_history.csv"
symbols = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"]
training_interval = 86400  

# ✅ 确保交易数据文件存在
if not os.path.exists(trade_history_file):
    pd.DataFrame(columns=["timestamp", "symbol", "action", "size", "price", "pnl"]).to_csv(trade_history_file, index=False)

# ✅ 获取市场数据（增加更多技术指标）
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

            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            df['rsi'] = 100 - (100 / (1 + gain / loss))

            df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()

            df['atr'] = df['high'].rolling(14).max() - df['low'].rolling(14).min()

            df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()

            df['bb_middle'] = df['close'].rolling(20).mean()
            df['bb_upper'] = df['bb_middle'] + 2 * df['close'].rolling(20).std()
            df['bb_lower'] = df['bb_middle'] - 2 * df['close'].rolling(20).std()

            df['adx'] = abs(df['high'] - df['low']).rolling(14).mean()

            df['stoch_rsi'] = (df['rsi'] - df['rsi'].rolling(14).min()) / (
                    df['rsi'].rolling(14).max() - df['rsi'].rolling(14).min())

            tp = (df['high'] + df['low'] + df['close']) / 3
            mean_tp = tp.rolling(20).mean()
            mean_dev = (tp - mean_tp).abs().rolling(20).mean()
            df['cci'] = (tp - mean_tp) / (0.015 * mean_dev)

            df = df.dropna()
            market_data[tf] = df

        return market_data
    except Exception as e:
        logging.error(f"⚠️ 获取市场数据失败: {e}")
        return None

# ✅ 计算交易信号
def get_trade_signal(symbol):
    data = get_market_data(symbol, timeframes=['5m', '1h', '1d'])
    if not data:
        return "hold"

    short_term, mid_term, long_term = data['5m'], data['1h'], data['1d']

    def calculate_signal(df):
        buy_votes, sell_votes = 0, 0

        if df['ma5'].iloc[-1] > df['ma15'].iloc[-1]: buy_votes += 1
        else: sell_votes += 1

        if df['rsi'].iloc[-1] < 30: buy_votes += 1
        elif df['rsi'].iloc[-1] > 70: sell_votes += 1

        if df['cci'].iloc[-1] < -100: buy_votes += 1
        elif df['cci'].iloc[-1] > 100: sell_votes += 1

        if df['stoch_rsi'].iloc[-1] < 0.2: buy_votes += 1
        elif df['stoch_rsi'].iloc[-1] > 0.8: sell_votes += 1

        if df['close'].iloc[-1] < df['bb_lower'].iloc[-1]: buy_votes += 1
        elif df['close'].iloc[-1] > df['bb_upper'].iloc[-1]: sell_votes += 1

        return buy_votes, sell_votes

    short_buy, short_sell = calculate_signal(short_term)
    mid_buy, mid_sell = calculate_signal(mid_term)
    long_buy, long_sell = calculate_signal(long_term)

    total_buy, total_sell = short_buy + mid_buy + long_buy, short_sell + mid_sell + long_sell

    if total_buy >= 7:
        return "buy"
    elif total_sell >= 7:
        return "sell"
    else:
        return "hold"

# ✅ 交易执行（止损 + 止盈）
def execute_trade(symbol, action, usdt_balance):
    try:
        leverage = get_dynamic_leverage(symbol)
        position_size = calculate_position_size(symbol, usdt_balance, leverage)

        exchange.set_leverage(leverage, symbol, params={"mgnMode": "isolated"})

        balance = exchange.fetch_balance()
        available_margin = balance['free'].get('USDT', 0)
        if available_margin < position_size * leverage:
            logging.warning(f"⚠️ 交易失败: 账户保证金不足 | 可用: {available_margin} USDT")
            return

        price = get_market_data(symbol)['5m']['close'].iloc[-1]
        stop_loss, take_profit = price * 0.98, price * 1.05

        exchange.create_order(symbol, "market", action, position_size, params={"stopLoss": stop_loss, "takeProfit": take_profit})
        logging.info(f"✅ 交易成功: {action.upper()} {position_size} 张 {symbol} | 杠杆: {leverage}x | 止损: {stop_loss} | 止盈: {take_profit}")
    
    except Exception as e:
        logging.error(f"⚠️ 交易失败: {e}")

# ✅ 交易机器人主循环
def trading_bot():
    last_training_time = time.time()

    while True:
        try:
            balance = exchange.fetch_balance()
            usdt_balance = balance['total'].get('USDT', 0)

            for symbol in symbols:
                signal = get_trade_signal(symbol)
                if signal in ["buy", "sell"]:
                    execute_trade(symbol, signal, usdt_balance)

            time.sleep(120)  
        
        except Exception as e:
            logging.error(f"⚠️ 交易循环错误: {e}")
            time.sleep(120)

# ✅ 启动机器人
trading_bot()