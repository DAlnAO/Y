import ccxt
import pandas as pd
import numpy as np
from datetime import datetime

# 配置信息（需替换为你的API信息）
config = {
    'apiKey': 'YOUR_API_KEY',
    'secret': 'YOUR_SECRET',
    'password': 'YOUR_PASSPHRASE',
    'options': {'defaultType': 'swap'}
}

# 初始化OKX连接
exchange = ccxt.okx(config)

def get_all_contracts():
    """获取所有USDT合约交易对"""
    markets = exchange.fetch_markets()
    return [m['symbol'] for m in markets if 'USDT' in m['symbol'] and 'SWAP' in m['id']]

def fetch_ohlcv_data(symbol):
    """获取多维历史数据"""
    try:
        # 获取1小时K线（最近24小时）
        ohlcv = exchange.fetch_ohlcv(symbol, '1h', limit=24)
        df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
        return df
    except Exception as e:
        print(f"获取{symbol}数据失败: {str(e)}")
        return None

def calculate_technical_factors(df):
    """技术指标计算"""
    # 动量因子
    df['momentum'] = df['close'].pct_change(periods=6) * 100  # 6小时动量
    
    # 波动率因子
    df['atr'] = df['high'] - df['low']
    volatility = df['atr'].mean() / df['close'].mean() * 100
    
    # 成交量因子
    volume_score = np.log1p(df['volume'].mean()) * 10  # 对数标准化
    
    return {
        'momentum': df['momentum'].iloc[-1],
        'volatility': volatility,
        'volume_score': volume_score
    }

def get_fundamental_factors(symbol):
    """基本面因子"""
    try:
        ticker = exchange.fetch_ticker(symbol)
        funding_rate = exchange.fetch_funding_rate(symbol)['fundingRate']
        
        return {
            'price_change': ticker['percentage'] * 100,  # 24小时涨跌幅
            'funding_rate': abs(funding_rate) * 10000,   # 资金费率(基点)
            'oi_change': exchange.fetch_open_interest_history(symbol)[-1]['openInterestValue']  # 持仓量变化
        }
    except:
        return {'price_change':0, 'funding_rate':0, 'oi_change':0}

def dynamic_scoring(symbol):
    """动态评分模型"""
    df = fetch_ohlcv_data(symbol)
    if df is None or len(df) < 24:
        return None
    
    tech_factors = calculate_technical_factors(df)
    fund_factors = get_fundamental_factors(symbol)
    
    # 合成评分权重
    score = (
        tech_factors['momentum'] * 0.4 + 
        tech_factors['volatility'] * 0.3 + 
        tech_factors['volume_score'] * 0.2 +
        fund_factors['price_change'] * 0.15 -
        fund_factors['funding_rate'] * 0.25 +
        np.log1p(fund_factors['oi_change']) * 0.1
    )
    
    return {
        'symbol': symbol,
        'score': round(score, 2),
        'momentum': tech_factors['momentum'],
        'volatility': tech_factors['volatility'],
        'funding_rate': fund_factors['funding_rate']
    }

def generate_strategy():
    """生成Top3币种策略"""
    contracts = get_all_contracts()
    ranked_coins = []
    
    for symbol in contracts:
        result = dynamic_scoring(symbol)
        if result and not np.isnan(result['score']):
            ranked_coins.append(result)
    
    # 按评分降序排列
    ranked_coins = sorted(ranked_coins, key=lambda x: x['score'], reverse=True)[:3]
    
    # 生成策略报告
    report = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'top3_coins': ranked_coins,
        'strategy_advice': {
            'momentum_strategy': "当短期动量>5%时开多仓，<-3%时平仓",
            'volatility_control': "波动率>15%的币种使用1%止损策略",
            'funding_arbitrage': "资金费率>30基点的币种建议反向操作"
        }
    }
    
    return report

if __name__ == "__main__":
    strategy = generate_strategy()
    print("=== OKX合约动态策略报告 ===")
    print(f"生成时间: {strategy['timestamp']}")
    print("\n推荐币种:")
    for coin in strategy['top3_coins']:
        print(f"{coin['symbol']} | 综合评分: {coin['score']} | 动量: {coin['momentum']:.2f}% | 波动率: {coin['volatility']:.2f}% | 资金费率: {coin['funding_rate']}基点")
    print("\n策略建议:")
    for k, v in strategy['strategy_advice'].items():
        print(f"- {k}: {v}")
