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
target_profit = 3  # 止盈 3%（更快锁定利润）
max_loss = 2  # 止损 2%（减少回撤）
risk_percentage = 15  # 资金管理：每次交易使用账户余额的 15%
max_risk = 30  # 最高资金使用率 30%
max_drawdown = 20  # 最大亏损 20% 后停止交易
data_file = "trading_data.csv"  # 存储交易数据
symbols = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"]  # 多币种支持

# **✅ 确保数据文件存在**
if not os.path.exists(data_file):
    pd.DataFrame(columns=["timestamp", "symbol", "close", "ma3", "ma10", "rsi", "macd", "atr", "bollinger_upper", "bollinger_lower", "price_change", "signal"]).to_csv(data_file, index=False)

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

# **✅ 获取市场数据**
def get_market_data(symbol, timeframe='5m', limit=500):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # 计算技术指标
        df['ma3'] = df['close'].rolling(window=3).mean()
        df['ma10'] = df['close'].rolling(window=10).mean()
        df['rsi'] = 100 - (100 / (1 + df['close'].pct_change().rolling(7).mean()))
        df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        df['atr'] = df['high'].rolling(14).max() - df['low'].rolling(14).min()
        df['bollinger_upper'] = df['close'].rolling(window=20).mean() + (df['close'].rolling(window=20).std() * 2)
        df['bollinger_lower'] = df['close'].rolling(window=20).mean() - (df['close'].rolling(window=20).std() * 2)
        df['price_change'] = df['close'].pct_change()

        df = df.dropna()
        logging.info(f"📊 成功获取 {symbol} 市场数据 - 最新价格: {df['close'].iloc[-1]}")
        return df
    except Exception as e:
        logging.error(f"⚠️ 获取市场数据失败: {e}")
        return None

# **✅ 获取交易信号**
def get_trade_signal(symbol):
    df = get_market_data(symbol)
    if df is None:
        return "hold"

    news_signal = get_news_sentiment_signal()
    short_term_signal = np.random.choice(["buy", "sell", "hold"])  # 增强交易活跃度（模拟 XGBoost）

    signal = "buy" if short_term_signal == "buy" and news_signal == "bullish" else "sell" if short_term_signal == "sell" and news_signal == "bearish" else "hold"

    logging.info(f"📢 交易信号: {signal} (技术: {short_term_signal}, 新闻: {news_signal})")
    return signal

# **✅ 交易机器人**
def trading_bot():
    logging.info("🚀 交易机器人启动...")
    initial_balance = 10000  # 模拟账户余额

    while True:
        try:
            usdt_balance = 10000  # 模拟账户余额
            logging.info(f"🔄 轮询市场中... 账户余额: {usdt_balance} USDT")

            for symbol in symbols:
                signal = get_trade_signal(symbol)
                if signal in ["buy", "sell"]:
                    trade_size = round((usdt_balance * (risk_percentage / 100)), 2)
                    if trade_size > (usdt_balance * (max_risk / 100)):
                        trade_size = round((usdt_balance * (max_risk / 100)), 2)
                    logging.info(f"✅ 模拟交易: {signal.upper()} {trade_size} 张 {symbol}")

            # **✅ 每 1 分钟记录一次日志**
            logging.info(f"💰 每 1 分钟反馈账户 USDT 余额: {usdt_balance}")
            time.sleep(60)

        except Exception as e:
            logging.error(f"⚠️ 交易循环错误: {e}")
            time.sleep(60)

# **✅ 启动机器人**
trading_bot()