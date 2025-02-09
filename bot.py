import ccxt
import time
import numpy as np

# OKX API 配置（请替换为你的 API Key）
API_KEY = "0f046e6a-1627-4db4-b97d-083d7e6cc16b"
API_SECRET = "BF7BC880C73AD54D2528FA271A358C2C"
API_PASSPHRASE = "Duan0918."

# 初始化 OKX 交易所（支持合约交易）
exchange = ccxt.okx({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'password': API_PASSPHRASE,
    'options': {'defaultType': 'swap'},  # 永续合约模式
})


# 获取账户余额（用于计算仓位）
def get_balance():
    balance = exchange.fetch_balance()
    usdt_balance = balance['total']['USDT']
    print(f"💰 账户 USDT 余额: {usdt_balance}")
    return usdt_balance


# 计算合适的杠杆（基于账户余额）
def select_leverage(balance):
    """根据账户余额自动选择杠杆"""
    if balance > 5000:
        return 3  # 资金大，使用低杠杆
    elif balance > 1000:
        return 5  # 资金中等，使用中等杠杆
    else:
        return 10  # 资金小，使用高杠杆


# 设置逐仓模式 & 自动杠杆
def set_margin_mode(symbol, balance):
    """ 设置逐仓模式并调整杠杆 """
    leverage = select_leverage(balance)
    params = {
        "instId": symbol,
        "lever": str(leverage),
        "mgnMode": "isolated"  # 逐仓模式
    }
    try:
        exchange.private_post_account_set_leverage(params)
        print(f"✅ 已设置 {symbol} 为逐仓模式，杠杆: {leverage}x")
    except Exception as e:
        print(f"⚠️ 设置杠杆失败: {e}")

    return leverage  # 需要保证 return 在 try-except 语句块之外


# 计算仓位大小（使用账户总仓位的 10%）
def calculate_position_size(balance, leverage):
    position_size = (balance * 0.1) * leverage  # 10% 余额 & 杠杆
    return round(position_size, 2)


# 获取市场数据（K线）
def get_market_data(symbol='ETH-USDT-SWAP', timeframe='15m', limit=50):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    closes = np.array([candle[4] for candle in ohlcv])  # 提取收盘价
    return closes


# 计算均线（用于趋势判断）
def moving_average(data, window=10):
    if len(data) < window:
        return None
    return np.mean(data[-window:])


# 交易逻辑（结合多时间框架 + 逐仓模式）
def trading_bot(symbol='ETH-USDT-SWAP'):
    usdt_balance = get_balance()
    leverage = set_margin_mode(symbol, usdt_balance)  # 自动选择杠杆
    position_size = calculate_position_size(usdt_balance, leverage)

    while True:
        # 获取多时间框架均线
        ma5m = moving_average(get_market_data(symbol, '5m'), 10)
        ma15m = moving_average(get_market_data(symbol, '15m'), 10)
        ma1h = moving_average(get_market_data(symbol, '1h'), 10)
        ma4h = moving_average(get_market_data(symbol, '4h'), 10)
        latest_price = get_market_data(symbol, '5m')[-1]  # 5分钟