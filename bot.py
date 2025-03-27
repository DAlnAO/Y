from flask import Flask, request, jsonify
import requests
import hmac
import base64
import time
import json

app = Flask(__name__)

# -------- 配置项 --------
API_KEY = 'ec02d4b0-8d79-4585-b3e9-609db23e06ad'
SECRET_KEY = 'BE8B7EE11F5E87FF922253F147E4680A'
PASSPHRASE = 'Duan0918@'

BASE_URL = 'https://www.okx.com'

def get_timestamp():
    return str(int(time.time()))

def sign(message, secret):
    mac = hmac.new(secret.encode(), message.encode(), digestmod='sha256')
    return base64.b64encode(mac.digest()).decode()

def place_order(symbol, side, price, size, order_type='limit'):
    url = f'{BASE_URL}/api/v5/trade/order'
    timestamp = get_timestamp()
    
    body = {
        "instId": symbol,
        "tdMode": "cross",
        "side": side,
        "ordType": order_type,
        "px": str(price),
        "sz": str(size)
    }

    message = f"{timestamp}POST/api/v5/trade/order{json.dumps(body)}"
    signature = sign(message, SECRET_KEY)

    headers = {
        'OK-ACCESS-KEY': API_KEY,
        'OK-ACCESS-SIGN': signature,
        'OK-ACCESS-TIMESTAMP': timestamp,
        'OK-ACCESS-PASSPHRASE': PASSPHRASE,
        'Content-Type': 'application/json'
    }

    response = requests.post(url, headers=headers, json=body)
    return response.json()

@app.route('/execute', methods=['POST'])
def execute():
    data = request.get_json()
    try:
        symbol = data['symbol']
        side = data['side']
        price = data['price']
        size = data['size']
        
        result = place_order(symbol, side, price, size)
        return jsonify({"status": "success", "response": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)