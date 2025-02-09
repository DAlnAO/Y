import os
import ccxt
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import time
import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
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
target_profit = 3  
max_loss = 2  
risk_percentage = 10  
max_risk = 30  
max_drawdown = 15  
cooldown_period = 600  # 10 分钟交易冷却时间
data_file = "trading_data.csv"
trade_history_file = "trade_history.csv"
symbols = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"]

# ✅ 确保数据文件存在
if not os.path.exists(data_file):
    pd.DataFrame(columns=["timestamp", "symbol", "close", "ma5", "ma15", "rsi", "macd", "atr", "obv", "price_change", "signal"]).to_csv(data_file, index=False)

if not os.path.exists(trade_history_file):
    pd.DataFrame(columns=["timestamp", "symbol", "action", "size", "price"]).to_csv(trade_history_file, index=False)

# ✅ 获取市场新闻
def fetch_market_news():
    url = "https://www.coindesk.com/"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        return [{"title": article.get_text().strip()} for article in soup.find_all("a", class_="headline")[:5]]
    except Exception as e:
        logging.error(f"⚠️ 获取市场新闻失败: {e}")
        return []

def analyze_news_sentiment(news_list):
    return sum(TextBlob(news["title"]).sentiment.polarity for news in news_list) / len(news_list) if news_list else 0

def get_news_sentiment_signal():
    score = analyze_news_sentiment(fetch_market_news())
    return "bullish" if score > 0.3 else "bearish" if score < -0.3 else "neutral"

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
        df['price_change'] = df['close'].pct_change()

        return df.dropna()
    except Exception as e:
        logging.error(f"⚠️ 获取市场数据失败: {e}")
        return None

# ✅ 训练 LSTM 模型
def train_lstm():
    df = pd.read_csv(data_file)
    if len(df) < 500:
        logging.warning("⚠️ 训练数据不足，跳过 LSTM 训练")
        return None

    scaler = MinMaxScaler()
    X = scaler.fit_transform(df[['ma5', 'ma15', 'rsi', 'macd', 'atr', 'obv', 'price_change']])
    X = X.reshape(-1, 7, 1)
    y = df['signal'].values

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(7,1)),
        Dropout(0.2),
        LSTM(64),
        BatchNormalization(),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=15, batch_size=32, verbose=1)
    
    return model

lstm_model = train_lstm()

# ✅ 交易逻辑
last_trade_time = {}

def get_trade_signal(symbol):
    df = get_market_data(symbol)
    if df is None:
        return "hold"

    news_signal = get_news_sentiment_signal()
    features = df[['ma5', 'ma15', 'rsi', 'macd', 'atr', 'obv', 'price_change']].values[-7:].reshape(1, 7, 1)
    lstm_signal = "buy" if lstm_model.predict(features)[0][0] > 0.5 else "sell"

    return "buy" if lstm_signal == "buy" and news_signal == "bullish" else "sell" if lstm_signal == "sell" and news_signal == "bearish" else "hold"

def execute_trade(symbol, action, size):
    try:
        exchange.create_market_order(symbol, action, size)
        trade_log = pd.DataFrame([{"timestamp": time.time(), "symbol": symbol, "action": action, "size": size}])
        trade_log.to_csv(trade_history_file, mode="a", header=False, index=False)
        last_trade_time[symbol] = time.time()
    except Exception as e:
        logging.error(f"⚠️ 交易失败: {e}")

# ✅ 交易机器人
def trading_bot():
    initial_balance = 10000

    while True:
        try:
            usdt_balance = 10000
            for symbol in symbols:
                if symbol in last_trade_time and time.time() - last_trade_time[symbol] < cooldown_period:
                    continue

                signal = get_trade_signal(symbol)
                if signal in ["buy", "sell"]:
                    trade_size = min(round((usdt_balance * (risk_percentage / 100)), 2), round((usdt_balance * (max_risk / 100)), 2))
                    execute_trade(symbol, signal, trade_size)

            if ((usdt_balance - initial_balance) / initial_balance) * 100 <= -max_drawdown:
                break

            time.sleep(60)

        except Exception as e:
            logging.error(f"⚠️ 交易循环错误: {e}")
            time.sleep(60)

trading_bot()