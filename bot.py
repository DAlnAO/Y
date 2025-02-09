import os
import ccxt
import pandas as pd
import numpy as np
import talib
import xgboost as xgb
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import time
import logging
from stable_baselines3 import DQN
import gym
from gym import spaces

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

# **添加技术指标**
def add_technical_indicators(df):
    df['ma5'] = talib.SMA(df['close'], timeperiod=5)
    df['ma15'] = talib.SMA(df['close'], timeperiod=15)
    df['ma50'] = talib.SMA(df['close'], timeperiod=50)
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['macd'], df['macd_signal'], _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
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
        print(f"⚠️ 获取持仓失败: {e}")
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

# **交易环境（强化学习）**
class TradingEnv(gym.Env):
    def __init__(self, symbol='ETH-USDT-SWAP', timeframe='15m', lookback=50):
        super(TradingEnv, self).__init__()

        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe
        self.lookback = lookback
        self.data = self.get_market_data()
        self.current_step = lookback
        self.balance = 10000
        self.position = 0

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(lookback, 5), dtype=np.float32)

    def get_market_data(self):
        df = get_market_data(self.symbol, self.timeframe, limit=1000)
        return df

    def step(self, action):
        prev_price = self.data.iloc[self.current_step - 1]['close']
        current_price = self.data.iloc[self.current_step]['close']

        if action == 0 and self.position == 0:
            self.position = self.balance / current_price
            self.balance = 0
        elif action == 1 and self.position > 0:
            self.balance = self.position * current_price
            self.position = 0

        new_balance = self.balance + (self.position * current_price)
        reward = new_balance - self.balance
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        obs = self.data.iloc[self.current_step - self.lookback:self.current_step].values

        return obs.reshape(1, -1), reward, done, {}

    def reset(self):
        self.current_step = self.lookback
        self.balance = 10000
        self.position = 0
        return self.data.iloc[self.current_step - self.lookback:self.current_step].values.reshape(1, -1)

# **训练 XGBoost 模型**
def train_xgboost():
    df = get_market_data('ETH-USDT-SWAP', '15m', 500)
    df = add_technical_indicators(df)
    
    X = df[['ma5', 'ma15', 'ma50', 'rsi', 'atr', 'macd']]
    y = (df['close'].shift(-1) > df['close']).astype(int)

    model_xgb = xgb.XGBClassifier()
    model_xgb.fit(X[:-1], y[:-1])

    return model_xgb

model_xgb = train_xgboost()

# **获取交易信号**
def get_trade_signal():
    df = get_market_data('ETH-USDT-SWAP', '15m', 500)
    df = add_technical_indicators(df)

    X = df[['ma5', 'ma15', 'ma50', 'rsi', 'atr', 'macd']]
    short_term_signal = model_xgb.predict(X[-1:])[0]

    news_signal = get_news_sentiment_signal()

    if short_term_signal == 1 and news_signal == "bullish":
        return "buy"
    elif short_term_signal == 0 and news_signal == "bearish":
        return "sell"
    else:
        return "hold"

# **交易机器人**
def trading_bot(symbol='ETH-USDT-SWAP'):
    while True:
        usdt_balance = get_balance()
        position = get_position(symbol)

        print(f"💰 账户 USDT 余额: {usdt_balance}")
        if position:
            print(f"📊 持仓: {position['side']} {position['size']} 张, 开仓价: {position['entry_price']}, 盈亏: {position['unrealized_pnl']}")
        else:
            print("📭 无持仓")

        signal = get_trade_signal()
        print(f"📢 交易信号: {signal}")

        time.sleep(60)

# **启动机器人**
trading_bot()