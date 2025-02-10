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
    'apiKey': "0f046e6a-1627-4db4-b97d-083d7e6cc16b",
    'secret': "BF7BC880C73AD54D2528FA271A358C2C",
    'password': "Duan0918.",
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

# ✅ 自动训练 LSTM 模型（如果没有模型文件）
def train_lstm_model(symbol):
    df = get_market_data(symbol, '5m')
    if df is None:
        logging.error(f"⚠️ 无法获取数据，无法训练模型: {symbol}")
        return None

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df['close'].values.reshape(-1, 1))

    X_train = []
    y_train = []
    look_back = 20
    for i in range(look_back, len(scaled_data)-1):
        X_train.append(scaled_data[i-look_back:i, 0])
        y_train.append(scaled_data[i+1, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = keras.Sequential([
        keras.layers.LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(50, return_sequences=False),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(25),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=16)

    # 保存模型
    model.save(f"lstm_model_{symbol}.h5")
    logging.info(f"✅ LSTM 模型训练完成并保存: {symbol}")
    return model, scaler

# ✅ 自动训练 XGBoost 模型（如果没有模型文件）
def train_xgb_model(symbol):
    df = get_market_data(symbol, '5m')
    if df is None:
        logging.error(f"⚠️ 无法获取数据，无法训练模型: {symbol}")
        return None

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df['close'].values.reshape(-1, 1))

    X_train = []
    y_train = []
    look_back = 20
    for i in range(look_back, len(scaled_data)-1):
        X_train.append(scaled_data[i-look_back:i, 0])
        y_train.append(scaled_data[i+1, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)

    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    xgb_model.fit(X_train, y_train)

    # 保存模型
    xgb_model.save_model(f"xgb_model_{symbol}.json")
    logging.info(f"✅ XGBoost 模型训练完成并保存: {symbol}")
    return xgb_model, scaler

# ✅ 使用 LSTM 和 XGBoost 模型进行预测
def predict_with_ai(symbol):
    df = get_market_data(symbol)
    if df is None:
        return None

    lstm_model_path = f"lstm_model_{symbol}.h5"
    xgb_model_path = f"xgb_model_{symbol}.json"

    if not os.path.exists(lstm_model_path):
        lstm_model, scaler = train_lstm_model(symbol)
    else:
        lstm_model = keras.models.load_model(lstm_model_path)
        scaler = MinMaxScaler()
        logging.info(f"✅ LSTM 模型加载成功: {symbol}")

    if not os.path.exists(xgb_model_path):
        xgb_model, scaler = train_xgb_model(symbol)
    else:
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model(xgb_model_path)
        logging.info(f"✅ XGBoost 模型加载成功: {symbol}")

    # 用模型进行预测
    scaled_data = scaler.fit_transform(df['close'].values.reshape(-1, 1))
    X_test = np.array(scaled_data[-20:]).reshape(1, -1, 1)

    lstm_pred = lstm_model.predict(X_test)
    xgb_pred = xgb_model.predict(X_test.reshape(1, -1))

    prediction = (lstm_pred + xgb_pred) / 2
    return prediction

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

    # 确保 prediction 是标量或可以被格式化的类型
    logging.info(f"📊 交易信号: {symbol} | 现价: {last_price:.2f} | LSTM: {lstm_pred:.2f} | XGB: {xgb_pred:.2f} | ARIMA: {arima_pred:.2f} | Transformer: {transformer_pred} | AI 预测: {prediction.item():.2f} | 信号: {signal.upper()}")
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