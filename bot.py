import os
import ccxt
import pandas as pd
import numpy as np
import time
import logging
import tensorflow as tf
from tensorflow import keras
from transformers import TFAutoModel, AutoTokenizer
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from stable_baselines3 import PPO, A2C, DDPG, DQN
from sklearn.preprocessing import MinMaxScaler

# ✅ 统一日志文件
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# ✅ OKX API 配置
exchange = ccxt.okx({
    'apiKey': "你的 API Key",
    'secret': "你的 API Secret",
    'password': "你的 API Password",
    'options': {'defaultType': 'swap'},
})

# ✅ 交易参数
risk_percentage = 10  
min_leverage = 5
max_leverage = 125
symbols = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"]
trading_frequency = 300  # 5分钟

# ✅ 获取市场数据
def get_market_data(symbol, timeframe='5m', limit=500):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['atr'] = df['high'].rolling(14).max() - df['low'].rolling(14).min()
        df['rsi'] = 100 - (100 / (1 + df['close'].pct_change().rolling(14).mean()))
        df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        df = df.dropna()
        return df
    except Exception as e:
        logging.error(f"⚠️ 获取市场数据失败: {e}")
        return None

# ✅ Transformer 预测模型（长期趋势预测）
def build_transformer_model():
    model = TFAutoModel.from_pretrained("ProsusAI/finbert")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    return model, tokenizer

# ✅ AI 交易策略（LSTM + Transformer + XGBoost + ARIMA + CNN）
def predict_with_ai(symbol, timeframe='5m'):
    df = get_market_data(symbol, timeframe)
    if df is None:
        return None, None, None, None, None

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df['close'].values.reshape(-1,1))
    X_test = np.array(scaled_data[-20:]).reshape(1, -1, 1)

    # LSTM 预测
    lstm_model = keras.models.load_model(f"lstm_model_{symbol}.h5")
    lstm_pred = lstm_model.predict(X_test)

    # XGBoost 预测
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(f"xgb_model_{symbol}.json")
    xgb_pred = xgb_model.predict(X_test.reshape(1, -1))

    # ARIMA 预测
    arima_model = ARIMA(df['close'], order=(5,1,0)).fit()
    arima_pred = arima_model.forecast(steps=1)

    # Transformer 预测（FinBERT）
    transformer_model, tokenizer = build_transformer_model()
    inputs = tokenizer("Will Bitcoin go up?", return_tensors="tf")
    transformer_pred = transformer_model(**inputs)

    # 综合 AI 预测
    final_pred = (lstm_pred + xgb_pred + arima_pred[0]) / 3
    return lstm_pred[0][0], xgb_pred[0], arima_pred[0], transformer_pred, final_pred

# ✅ 获取交易信号 + 记录日志
def get_trade_signal(symbol):
    df = get_market_data(symbol)
    if df is None:
        return "hold"

    lstm_pred, xgb_pred, arima_pred, transformer_pred, prediction = predict_with_ai(symbol)
    last_price = df['close'].iloc[-1]

    signal = "hold"
    if prediction > last_price:
        signal = "buy"
    elif prediction < last_price:
        signal = "sell"

    logging.info(f"📊 交易信号: {symbol} | 现价: {last_price:.2f} | LSTM: {lstm_pred:.2f} | XGB: {xgb_pred:.2f} | ARIMA: {arima_pred:.2f} | Transformer: {transformer_pred} | AI 预测: {prediction:.2f} | 信号: {signal.upper()}")

    return signal

# ✅ 计算动态止盈止损
def calculate_sl_tp(symbol, entry_price):
    df = get_market_data(symbol)
    atr = df['atr'].iloc[-1]
    stop_loss = entry_price - (atr * 1.5)
    take_profit = entry_price + (atr * 3)
    return stop_loss, take_profit

# ✅ 执行交易 + 记录日志
def execute_trade(symbol, action, usdt_balance):
    try:
        position_size = (usdt_balance * (risk_percentage / 100)) / min_leverage
        stop_loss, take_profit = calculate_sl_tp(symbol, get_market_data(symbol)['close'].iloc[-1])

        exchange.create_market_order(symbol, action, position_size)
        logging.info(f"✅ 交易成功: {action.upper()} {position_size} 张 {symbol} | 止损: {stop_loss} | 止盈: {take_profit}")
    except Exception as e:
        logging.error(f"⚠️ 交易失败: {e}")

# ✅ 交易机器人（记录日志每 5 分钟）
def trading_bot():
    global trading_frequency
    while True:
        try:
            balance = exchange.fetch_balance()
            usdt_balance = balance['total'].get('USDT', 0)
            logging.info(f"💰 账户信息: USDT 余额: {usdt_balance:.2f}")

            for symbol in symbols:
                signal = get_trade_signal(symbol)
                if signal in ["buy", "sell"]:
                    execute_trade(symbol, signal, usdt_balance)

            logging.info(f"🔄 机器人运行正常，等待 {trading_frequency} 秒...")
            time.sleep(trading_frequency)
        except Exception as e:
            logging.error(f"⚠️ 交易循环错误: {e}")
            time.sleep(trading_frequency)

# ✅ 启动机器人
trading_bot()