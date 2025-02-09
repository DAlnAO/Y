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
    'apiKey': "你的API_KEY",
    'secret': "你的API_SECRET",
    'password': "你的API_PASSPHRASE",
    'options': {'defaultType': 'swap'},
})

# **获取市场数据**
def get_market_data(symbol='ETH-USDT-SWAP', timeframe='15m', limit=500):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# **查询账户 USDT 余额**
def get_balance():
    balance = exchange.fetch_balance()
    return balance['total']['USDT']

# **查询合约持仓状态**
def get_position(symbol):
    try:
        positions = exchange.fetch_positions()
        for pos in positions:
            if pos['symbol'] == symbol and abs(pos['contracts']) > 0:
                return {
                    "side": pos['side'],
                    "size": pos['contracts'],
                    "entry_price": pos['entryPrice'],
                    "unrealized_pnl": pos['unrealizedPnl']
                }
    except Exception as e:
        logging.error(f"⚠️ 获取持仓失败: {e}")
    return None

# **获取市场新闻**
def fetch_market_news():
    url = "https://www.coindesk.com/"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    news_list = []
    for article in soup.find_all("a", class_="headline"):
        title = article.get_text().strip()
        link = article["href"]
        news_list.append({"title": title, "link": link})
    
    return news_list[:5]

# **计算新闻情绪**
def analyze_news_sentiment(news_list):
    sentiment_score = 0
    for news in news_list:
        analysis = TextBlob(news["title"])
        sentiment_score += analysis.sentiment.polarity

    return sentiment_score / len(news_list)

# **获取新闻情绪信号**
def get_news_sentiment_signal():
    news_list = fetch_market_news()
    sentiment_score = analyze_news_sentiment(news_list)

    if sentiment_score > 0.3:
        return "bullish"
    elif sentiment_score < -0.3:
        return "bearish"
    else:
        return "neutral"

# **训练 XGBoost 模型（去掉 talib 指标）**
def train_xgboost():
    df = get_market_data('ETH-USDT-SWAP', '15m', 500)

    # 使用简单的均值和变化率作为特征
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma15'] = df['close'].rolling(window=15).mean()
    df['price_change'] = df['close'].pct_change()

    X = df[['ma5', 'ma15', 'price_change']].dropna()
    y = (df['close'].shift(-1) > df['close']).astype(int)[X.index]

    model_xgb = xgb.XGBClassifier()
    model_xgb.fit(X, y)

    return model_xgb

model_xgb = train_xgboost()

# **获取交易信号**
def get_trade_signal():
    df = get_market_data('ETH-USDT-SWAP', '15m', 500)

    # 计算均值和变化率
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma15'] = df['close'].rolling(window=15).mean()
    df['price_change'] = df['close'].pct_change()

    X = df[['ma5', 'ma15', 'price_change']].dropna()

    short_term_signal = model_xgb.predict(X[-1:])[0]
    news_signal = get_news_sentiment_signal()

    if short_term_signal == 1 and news_signal == "bullish":
        return "buy"
    elif short_term_signal == 0 and news_signal == "bearish":
        return "sell"
    else:
        return "hold"

# **执行交易**
def execute_trade(symbol, action, size):
    try:
        exchange.create_market_order(symbol, action, size)
        logging.info(f"✅ 交易执行: {action.upper()} {size} 张 {symbol}")
    except Exception as e:
        logging.error(f"⚠️ 交易执行失败: {e}")

# **交易机器人**
def trading_bot(symbol='ETH-USDT-SWAP'):
    while True:
        try:
            usdt_balance = get_balance()
            position = get_position(symbol)

            logging.info(f"💰 账户 USDT 余额: {usdt_balance}")
            if position:
                logging.info(f"📊 持仓: {position['side']} {position['size']} 张, 开仓价: {position['entry_price']}, 盈亏: {position['unrealized_pnl']}")
            else:
                logging.info("📭 无持仓")

            signal = get_trade_signal()
            logging.info(f"📢 交易信号: {signal}")

            if signal == "buy" and not position:
                execute_trade(symbol, "buy", 10)
            elif signal == "sell" and not position:
                execute_trade(symbol, "sell", 10)

            time.sleep(60)

        except Exception as e:
            logging.error(f"⚠️ 交易循环错误: {e}")
            time.sleep(10)

# **启动机器人**
trading_bot()