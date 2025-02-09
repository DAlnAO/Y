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

# **执行交易**
def execute_trade(symbol, action, size):
    try:
        exchange.create_market_order(symbol, action, size)
        logging.info(f"✅ 交易执行: {action.upper()} {size} 张 {symbol}")
    except Exception as e:
        logging.error(f"⚠️ 交易执行失败: {e}")

# **止盈止损检查**
def check_take_profit_stop_loss(symbol, position):
    if not position:
        return False  # 没有持仓，不需要检查

    entry_price = position["entry_price"]
    unrealized_pnl = position["unrealized_pnl"]

    # 计算盈利/亏损百分比
    profit_loss_percent = (unrealized_pnl / entry_price) * 100

    if profit_loss_percent >= target_profit:
        logging.info(f"🎯 触发止盈: {profit_loss_percent:.2f}% 平仓")
        execute_trade(symbol, "sell" if position["side"] == "long" else "buy", position["size"])
        return True

    if profit_loss_percent <= -max_loss:
        logging.info(f"⛔ 触发止损: {profit_loss_percent:.2f}% 平仓")
        execute_trade(symbol, "sell" if position["side"] == "long" else "buy", position["size"])
        return True

    return False

# **交易机器人**
def trading_bot(symbol='ETH-USDT-SWAP'):
    initial_balance = get_balance()  # 记录初始资金

    while True:
        try:
            usdt_balance = get_balance()
            position = get_position(symbol)

            # **✅ 每 30 秒记录账户信息**
            logging.info(f"💰 账户 USDT 余额: {usdt_balance}")

            if position:
                logging.info(f"📊 当前持仓: {position}")

                # **✅ 检查止盈止损**
                if check_take_profit_stop_loss(symbol, position):
                    continue

            # **✅ 风险控制：如果亏损超过 `max_drawdown`% 停止交易**
            if (usdt_balance / initial_balance - 1) * 100 <= -max_drawdown:
                logging.warning("⚠️ 账户亏损超出限制，停止交易！")
                break

            # **✅ 获取交易信号**
            signal = get_trade_signal()
            logging.info(f"📢 交易信号: {signal}")

            if signal in ["buy", "sell"] and not position:
                trade_size = round((usdt_balance * (risk_percentage / 100)), 2)
                execute_trade(symbol, signal, trade_size)

            # **✅ 30 秒后继续循环**
            time.sleep(30)

        except Exception as e:
            logging.error(f"⚠️ 交易循环错误: {e}")
            time.sleep(10)

# **启动机器人**
trading_bot()