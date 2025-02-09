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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

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
        news_list = [{"title": article.get_text().strip()} for article in soup.find_all("a", class_="headline")[:5]]
        logging.info(f"📰 成功获取市场新闻: {news_list[:3]}")
        return news_list
    except Exception as e:
        logging.error(f"⚠️ 获取市场新闻失败: {e}")
        return []

def analyze_news_sentiment(news_list):
    return sum(TextBlob(news["title"]).sentiment.polarity for news in news_list) / len(news_list) if news_list else 0

def get_news_sentiment_signal():
    score = analyze_news_sentiment(fetch_market_news())
    signal = "bullish" if score > 0.3 else "bearish" if score < -0.3 else "neutral"
    logging.info(f"📊 新闻情绪信号: {signal}")
    return signal

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

        df = df.dropna()
        logging.info(f"📊 获取 {symbol} 市场数据成功，最新价格: {df['close'].iloc[-1]}")
        return df
    except Exception as e:
        logging.error(f"⚠️ 获取市场数据失败: {e}")
        return None

# ✅ 训练 LSTM 模型
def train_lstm():
    try:
        df = pd.read_csv(data_file, on_bad_lines='skip')
        if len(df) < 500:
            logging.warning("⚠️ 训练数据不足，LSTM 训练跳过")
            return None

        X = df[['ma5', 'ma15', 'rsi', 'macd', 'atr', 'obv', 'price_change']].values.reshape(-1, 7, 1)
        y = df['signal'].values

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(7,1)),
            LSTM(50),
            Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X, y, epochs=10, batch_size=16, verbose=0)

        logging.info("✅ LSTM 训练完成")
        return model
    except Exception as e:
        logging.error(f"⚠️ LSTM 训练失败: {e}")
        return None

lstm_model = train_lstm()

# ✅ 交易逻辑
last_trade_time = {}

def get_trade_signal(symbol):
    df = get_market_data(symbol)
    if df is None or len(df) < 7:
        logging.warning(f"⚠️ {symbol} 交易信号计算失败: 数据不足 7 行")
        return "hold"

    news_signal = get_news_sentiment_signal()
    features = df[['ma5', 'ma15', 'rsi', 'macd', 'atr', 'obv', 'price_change']].values[-7:]
    
    if lstm_model:
        lstm_prediction = lstm_model.predict(features.reshape(1, 7, 1))[0][0]
        lstm_signal = "buy" if lstm_prediction > 0.5 else "sell"
    else:
        lstm_signal = np.random.choice(["buy", "sell", "hold"])

    signal = "buy" if lstm_signal == "buy" and news_signal == "bullish" else "sell" if lstm_signal == "sell" and news_signal == "bearish" else "hold"
    
    logging.info(f"📢 交易信号: {signal} (LSTM: {lstm_signal}, 新闻: {news_signal})")
    return signal

def execute_trade(symbol, action, size):
    for _ in range(3):
        try:
            exchange.create_market_order(symbol, action, size)
            logging.info(f"✅ 交易成功: {action.upper()} {size} 张 {symbol}")
            last_trade_time[symbol] = time.time()
            return
        except Exception as e:
            logging.error(f"⚠️ 交易失败: {e}")
            time.sleep(2)

# ✅ 交易机器人
def trading_bot():
    logging.info("🚀 交易机器人启动...")
    initial_balance = 10000
    
    while True:
        try:
            usdt_balance = 10000
            for symbol in symbols:
                if symbol in last_trade_time and time.time() - last_trade_time[symbol] < cooldown_period:
                    continue

                signal = get_trade_signal(symbol)
                if signal in ["buy", "sell"]:
                    trade_size = round((usdt_balance * (risk_percentage / 100)), 2)
                    execute_trade(symbol, signal, trade_size)

            if ((usdt_balance - initial_balance) / initial_balance) * 100 <= -max_drawdown:
                break

            logging.info(f"💰 每 1 分钟记录 USDT 余额: {usdt_balance}")
            time.sleep(60)

        except Exception as e:
            logging.error(f"⚠️ 交易循环错误: {e}")
            time.sleep(60)

trading_bot()