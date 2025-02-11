import ccxt
import gym
import numpy as np
import pandas as pd
import time
import logging
import threading
from stable_baselines3 import PPO
from collections import deque

# ✅ 统一日志文件
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# ✅ OKX API 配置
exchange = ccxt.okx({
    'apiKey': "your_api_key",
    'secret': "your_secret_key",
    'password': "your_api_password",
    'options': {'defaultType': 'swap'},
})

# ✅ 交易参数
symbols = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP", "ADA-USDT-SWAP", "BNB-USDT-SWAP"]
risk_percentage = 10  # 风险百分比（每次交易的最大风险）
max_position_percentage = 50  # 最大仓位占比
trading_frequency = 300  # 每5分钟交易一次
training_interval = 1800  # 每30分钟继续训练一次
stop_loss_percentage = 5  # 止损百分比
take_profit_percentage = 10  # 止盈百分比

# 交易记录列表
trade_history = []

# ✅ 获取市场数据
def get_market_data(symbol, timeframes=['5m'], limit=500):
    try:
        market_data = {}
        for tf in timeframes:
            ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['ma5'] = df['close'].rolling(5).mean()
            df['ma15'] = df['close'].rolling(15).mean()
            df['atr'] = df['high'].rolling(14).max() - df['low'].rolling(14).min()
            df['rsi'] = 100 - (100 / (1 + df['close'].pct_change().rolling(14).mean()))
            df = df.dropna()
            market_data[tf] = df
        return market_data
    except Exception as e:
        logging.error(f"⚠️ 获取市场数据失败: {symbol} | {e}")
        return None

# ✅ PPO 强化学习交易环境
class TradingEnv(gym.Env):
    def __init__(self, symbol, window_size=20):
        super(TradingEnv, self).__init__()
        self.symbol = symbol
        self.window_size = window_size
        self.action_space = gym.spaces.Discrete(3)  # 3个动作：买入，卖出，持有
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(window_size, 4), dtype=np.float32)  # 每次输入窗口大小的状态
        self.market_data = get_market_data(symbol)['5m']
        self.current_step = window_size
        self.balance = 1000  # 初始余额1000 USDT
        self.position = 0  # 初始无仓位
        self.entry_price = 0  # 初始化买入价格
        self.history = deque(maxlen=window_size)  # 保存最近的窗口数据

    def step(self, action):
        current_price = self.market_data['close'].iloc[self.current_step]
        reward = 0

        # 买入操作
        if action == 1 and self.balance > 0:
            self.position = self.balance / current_price
            self.balance = 0
            self.entry_price = current_price  # 记录买入价格
        # 卖出操作
        elif action == 2 and self.position > 0:
            self.balance = self.position * current_price
            self.position = 0
            reward = self.balance - 1000  # 奖励是利润
            self.entry_price = 0  # 清空买入价格

        self.current_step += 1
        done = self.current_step >= len(self.market_data) - 1
        state = self._get_state()

        # 止盈止损检查
        if self.position > 0:
            price_change_percentage = (current_price - self.entry_price) / self.entry_price * 100
            if price_change_percentage >= take_profit_percentage:  # 达到止盈条件
                logging.info(f"✅ 止盈: 当前价格达到止盈点 {take_profit_percentage}%")
                self.balance = self.position * current_price
                self.position = 0
                reward = self.balance - 1000  # 奖励是利润
            elif price_change_percentage <= -stop_loss_percentage:  # 达到止损条件
                logging.info(f"⚠️ 止损: 当前价格达到止损点 {-stop_loss_percentage}%")
                self.balance = self.position * current_price
                self.position = 0
                reward = self.balance - 1000  # 奖励是损失

        return state, reward, done, {}

    def reset(self):
        self.current_step = self.window_size
        self.balance = 1000
        self.position = 0
        self.entry_price = 0
        self.history.clear()
        return self._get_state()

    def _get_state(self):
        # 返回窗口大小内的数据
        window_data = np.array([[
            self.market_data['ma5'].iloc[i],
            self.market_data['ma15'].iloc[i],
            self.market_data['atr'].iloc[i],
            self.market_data['rsi'].iloc[i]
        ] for i in range(self.current_step - self.window_size, self.current_step)])
        return window_data

# ✅ 计算最大可用仓位
def calculate_position_size(usdt_balance, max_position_percentage):
    available_balance = usdt_balance * (max_position_percentage / 100)  # 最大仓位限制
    return available_balance

# ✅ 训练 PPO 交易代理
def train_ppo_agent(symbol):
    env = TradingEnv(symbol)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=50000)  # 可以适当增大训练时长
    model.save(f"ppo_trading_agent_{symbol}")
    logging.info(f"✅ {symbol} PPO 训练完成")

# ✅ 交易信号预测（PPO）
def get_trade_signal(symbol):
    try:
        ppo_model = PPO.load(f"ppo_trading_agent_{symbol}")
        data = get_market_data(symbol, timeframes=['5m'])
        if not data:
            return "hold"

        df = data['5m']
        state = np.array([
            df['ma5'].iloc[-1],
            df['ma15'].iloc[-1],
            df['atr'].iloc[-1],
            df['rsi'].iloc[-1]
        ])

        ppo_action, _ = ppo_model.predict(state)
        return "buy" if ppo_action == 1 else "sell"
    except Exception as e:
        logging.error(f"⚠️ 交易信号错误: {e}")
        return "hold"

# ✅ 交易执行
def execute_trade(symbol, action, usdt_balance):
    try:
        # 计算仓位大小
        position_size = calculate_position_size(usdt_balance, max_position_percentage)
        
        if action == "buy" and position_size > 0:
            exchange.create_market_order(symbol, action, position_size, {'marginMode': 'isolated'})  # 使用逐仓模式
            logging.info(f"✅ 交易成功: {action.upper()} {position_size} {symbol}")

        elif action == "sell" and position_size > 0:
            exchange.create_market_order(symbol, action, position_size, {'marginMode': 'isolated'})
            logging.info(f"✅ 交易成功: {action.upper()} {position_size} {symbol}")

        # 记录交易行为
        trade_info = {
            "symbol": symbol,
            "action": action,
            "position_size": position_size,
            "price": exchange.fetch_ticker(symbol)['last'],
            "timestamp": time.time()
        }
        trade_history.append(trade_info)
    except Exception as e:
        logging.error(f"⚠️ 交易失败: {e}")

# ✅ 定期反馈交易历史
def feedback_to_log():
    while True:
        time.sleep(300)  # 每5分钟反馈一次
        # 反馈市场数据
        logging.info("⏱️ 5分钟内的市场数据反馈:")
        for symbol in symbols:
            market_data = get_market_data(symbol)
            if market_data is not None:
                latest_data = market_data['5m'].iloc[-1]
                logging.info(f"📈 {symbol} 当前价格: {latest_data['close']} | MA5: {latest_data['ma5']} | MA15: {latest_data['ma15']} | ATR: {latest_data['atr']} | RSI: {latest_data['rsi']}")

        # 反馈交易信号预测
        logging.info("⏱️ 5分钟内的交易信号预测: ")
        for symbol in symbols:
            signal = get_trade_signal(symbol)
            logging.info(f"🔮 {symbol} 当前信号: {signal}")

        # 反馈交易执行情况
        logging.info("⏱️ 5分钟内的交易执行情况: ")
        if trade_history:
            for trade in trade_history:
                logging.info(f"💰 交易记录: {trade}")
        else:
            logging.info("没有新的交易行为")
        
        trade_history.clear()  # 清空交易记录

# ✅ 交易机器人
def trading_bot():
    while True:
        try:
            usdt_balance = exchange.fetch_balance()['total'].get('USDT', 0)
            for symbol in symbols:
                signal = get_trade_signal(symbol)
                if signal in ["buy", "sell"]:
                    execute_trade(symbol, signal, usdt_balance)
            time.sleep(trading_frequency)
        except Exception as e:
            logging.error(f"⚠️ 交易失败: {e}")

# 启动线程
feedback_thread = threading.Thread(target=feedback_to_log)
feedback_thread.daemon = True
feedback_thread.start()

# 启动交易机器人
trading_bot()