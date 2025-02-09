import ccxt
import time
import logging
import numpy as np

# **初始化日志**
logging.basicConfig(filename="trading_bot.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# **OKX API 配置（替换为你的 API Key）**
API_KEY = "0f046e6a-1627-4db4-b97d-083d7e6cc16b"
API_SECRET = "BF7BC880C73AD54D2528FA271A358C2C"
API_PASSPHRASE = "Duan0918."

exchange = ccxt.okx({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'password': API_PASSPHRASE,
    'options': {'defaultType': 'swap'},
})

# **交易参数**
max_drawdown = 5  # 最大回撤 5%
risk_percentage = 2  # 每次交易占账户资金的 2%
take_profit_ratio = 1.05  # 止盈 5%
stop_loss_ratio = 0.95  # 止损 5%


# **✅ 获取账户余额**
def get_balance():
    try:
        balance = exchange.fetch_balance()
        usdt_balance = balance['total']['USDT']
        return usdt_balance
    except Exception as e:
        logging.error(f"⚠️ 获取账户余额失败: {e}")
        return 0


# **✅ 获取市场数据**
def get_market_data(symbol='ETH-USDT-SWAP', timeframe='15m', limit=50):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        closes = np.array([candle[4] for candle in ohlcv])  # 提取收盘价
        return closes
    except Exception as e:
        logging.error(f"⚠️ 获取市场数据失败: {e}")
        return np.array([])


# **✅ 计算均线**
def moving_average(data, window=10):
    if len(data) < window:
        return None
    return np.mean(data[-window:])


# **✅ 获取交易信号**
def get_trade_signal(symbol="ETH-USDT-SWAP"):
    try:
        data = get_market_data(symbol)
        if len(data) == 0:
            return "hold"

        ma10 = moving_average(data, 10)
        ma50 = moving_average(data, 50)

        if ma10 > ma50:
            return "buy"
        elif ma10 < ma50:
            return "sell"
        else:
            return "hold"
    except Exception as e:
        logging.error(f"⚠️ 计算交易信号失败: {e}")
        return "hold"


# **✅ 获取当前持仓**
def get_position(symbol):
    try:
        positions = exchange.fetch_positions()
        for position in positions:
            if position['symbol'] == symbol and position['contracts'] > 0:
                return position
        return None
    except Exception as e:
        logging.error(f"⚠️ 获取持仓失败: {e}")
        return None


# **✅ 执行交易**
def execute_trade(symbol, side, amount):
    try:
        order = exchange.create_market_order(symbol, side, amount)
        logging.info(f"✅ 交易执行成功: {order}")
    except Exception as e:
        logging.error(f"⚠️ 交易执行失败: {e}")


# **✅ 检查止盈止损**
def check_take_profit_stop_loss(symbol, position):
    entry_price = position["entryPrice"]
    current_price = exchange.fetch_ticker(symbol)['last']

    take_profit_price = entry_price * take_profit_ratio
    stop_loss_price = entry_price * stop_loss_ratio

    if current_price >= take_profit_price:
        logging.info(f"🎯 触发止盈: {current_price} 平仓")
        execute_trade(symbol, "sell" if position["side"] == "long" else "buy", position["size"])
        return True
    elif current_price <= stop_loss_price:
        logging.info(f"⛔ 触发止损: {current_price} 平仓")
        execute_trade(symbol, "sell" if position["side"] == "long" else "buy", position["size"])
        return True

    return False


# **✅ 交易机器人**
def trading_bot(symbol='ETH-USDT-SWAP'):
    initial_balance = get_balance()
    
    while True:
        try:
            usdt_balance = get_balance()
            position = get_position(symbol)

            # **✅ 记录账户信息**
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


# **✅ 启动机器人**
trading_bot()