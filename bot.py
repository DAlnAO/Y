import os
import ccxt
import pandas as pd
import numpy as np
import xgboost as xgb
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import time
import logging

# **日志系统**
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# **OKX API 配置**
exchange = ccxt.okx({
    'apiKey': "0f046e6a-1627-4db4-b97d-083d7e6cc16b",
    'secret': "BF7BC880C73AD54D2528FA271A358C2C",
    'password': "Duan0918.",
    'options': {'defaultType': 'swap'},
})

# **参数设置**
target_profit = 5  # 止盈 5%
max_loss = 3  # 止损 3%
risk_percentage = 10  # 资金管理：每次交易使用账户余额的 10%
max_drawdown = 20  # 最大亏损 20% 后停止交易
data_file = "trading_data.csv"  # 存储交易数据

# **✅ 确保数据文件存在**
if not os.path.exists(data_file):
    pd.DataFrame(columns=["timestamp", "close", "ma5", "ma15", "price_change", "signal"]).to_csv(data_file, index=False)

# **✅ 获取市场数据**
def get_market_data(symbol='ETH-USDT-SWAP', timeframe='15m', limit=500):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        logging.info(f"📊 成功获取 {symbol} 市场数据 - 最新价格: {df['close'].iloc[-1]}")
        return df
    except Exception as e:
        logging.error(f"⚠️ 获取市场数据失败: {e}")
        return None

# **✅ 获取市场新闻**
def fetch_market_news():
    url = "https://www.coindesk.com/"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        news_list = []
        for article in soup.find_all("a", class_="headline"):
            title = article.get_text().strip()
            link = article["href"]
            news_list.append({"title": title, "link": link})

        logging.info(f"📰 成功获取市场新闻: {news_list[:3]}")
        return news_list[:5]
    except Exception as e:
        logging.error(f"⚠️ 获取市场新闻失败: {e}")
        return []

# **✅ 计算新闻情绪**
def analyze_news_sentiment(news_list):
    if not news_list:
        return 0  # 如果无法获取新闻，默认情绪为中性

    sentiment_score = sum(TextBlob(news["title"]).sentiment.polarity for news in news_list)
    score = sentiment_score / len(news_list)
    logging.info(f"📊 新闻情绪得分: {score}")
    return score

# **✅ 获取新闻情绪信号**
def get_news_sentiment_signal():
    news_list = fetch_market_news()
    sentiment_score = analyze_news_sentiment(news_list)

    if sentiment_score > 0.3:
        return "bullish"
    elif sentiment_score < -0.3:
        return "bearish"
    else:
        return "neutral"

# **✅ 训练 XGBoost 模型**
def train_xgboost():
    try:
        df = pd.read_csv(data_file)
        if len(df) < 20:
            logging.error("⚠️ XGBoost 训练失败：数据不足")
            return None

        X = df[['ma5', 'ma15', 'price_change']]
        y = df['signal']

        model_xgb = xgb.XGBClassifier()
        model_xgb.fit(X, y)
        logging.info("✅ XGBoost 训练完成")
        return model_xgb
    except Exception as e:
        logging.error(f"⚠️ XGBoost 训练失败: {e}")
        return None

model_xgb = train_xgboost()

# **✅ 获取交易信号**
def get_trade_signal():
    df = get_market_data('ETH-USDT-SWAP', '15m', 500)
    if df is None:
        return "hold"

    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma15'] = df['close'].rolling(window=15).mean()
    df['price_change'] = df['close'].pct_change()
    df = df.dropna()

    latest_data = df.iloc[-1][['timestamp', 'close', 'ma5', 'ma15', 'price_change']].to_dict()

    news_signal = get_news_sentiment_signal()
    short_term_signal = 0 if model_xgb is None else model_xgb.predict(df[['ma5', 'ma15', 'price_change']][-1:])[0]

    signal = "buy" if short_term_signal == 1 and news_signal == "bullish" else "sell" if short_term_signal == 0 and news_signal == "bearish" else "hold"

    latest_data["signal"] = 1 if signal == "buy" else 0 if signal == "sell" else -1
    pd.DataFrame([latest_data]).to_csv(data_file, mode='a', header=False, index=False)

    logging.info(f"📢 交易信号: {signal} (XGBoost: {'BUY' if short_term_signal == 1 else 'SELL'}, 新闻信号: {news_signal})")
    return signal

# **✅ 获取账户余额**
def get_balance():
    try:
        balance = exchange.fetch_balance()
        usdt_balance = balance['total'].get('USDT', 0)
        logging.info(f"💰 账户 USDT 余额: {usdt_balance}")
        return usdt_balance
    except Exception as e:
        logging.error(f"⚠️ 获取账户余额失败: {e}")
        return 0

# **✅ 执行交易**
def execute_trade(symbol, action, size):
    try:
        order = exchange.create_market_order(symbol, action, size)
        logging.info(f"✅ 交易执行成功: {action.upper()} {size} 张 {symbol} - 订单详情: {order}")
    except Exception as e:
        logging.error(f"⚠️ 交易执行失败: {e}")

# **✅ 交易机器人**
def trading_bot(symbol='ETH-USDT-SWAP'):
    logging.info("🚀 交易机器人启动...")
    initial_balance = get_balance()

    while True:
        try:
            usdt_balance = get_balance()
            logging.info(f"🔄 轮询市场中... 账户余额: {usdt_balance} USDT")

            # **✅ 获取交易信号**
            signal = get_trade_signal()
            logging.info(f"📢 交易信号: {signal}")

            if signal in ["buy", "sell"]:
                trade_size = round((usdt_balance * (risk_percentage / 100)), 2)
                execute_trade(symbol, signal, trade_size)

            # **✅ 每 30 秒反馈账户 USDT 余额**
            logging.info(f"💰 每 30 秒反馈账户 USDT 余额: {usdt_balance}")
            time.sleep(30)

        except Exception as e:
            logging.error(f"⚠️ 交易循环错误: {e}")
            time.sleep(10)

# **✅ 启动机器人**
trading_bot()