import ccxt

exchange = ccxt.okx({
    'apiKey': "7963fab7-9c36-4ef7-8dd8-9d803f850ba0",
    'secret': "D67ECA3298D061A006BFF4DE3EFB2583",
    'password': "Duan0918.",
    'options': {'defaultType': 'swap'},
    # 'sandboxMode': True  # ✅ 如果是模拟交易，打开这个
})

try:
    balance = exchange.fetch_balance()
    print(balance)  # ✅ 如果这里能获取余额，API Key 是正确的
except Exception as e:
    print(f"API 连接失败: {e}")  # ❌ 如果这里报错，说明 API Key 有问题