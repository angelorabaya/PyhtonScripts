import requests
import pandas as pd
import time
import hmac
import hashlib

# Replace with your API key and secret
API_KEY = 'oNvvtFX7hbQrG5eqij'
API_SECRET = 'kl9d7XCEFt2V6AvqOGru4TEozivsP8vzeJ28'

# Function to create the signature for the request
def sign_request(api_key, api_secret, params):
    params['api_key'] = api_key
    # Sort parameters alphabetically
    sorted_params = sorted(params.items())
    # Stringify and concatenate sorted params
    params_string = '&'.join([f"{key}={value}" for key, value in sorted_params])
    # Create the signature using HMAC SHA256
    signature = hmac.new(api_secret.encode(), params_string.encode(), hashlib.sha256).hexdigest()
    return signature

# Fetch historical data
def get_historical_data(symbol, interval, limit=200):
    endpoint = "https://api.bybit.com/v2/public/kline/list"
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit,
        'timestamp': int(time.time() * 1000)
    }

    # Sign the request
    signature = sign_request(API_KEY, API_SECRET, params)
    params['sign'] = signature

    response = requests.get(endpoint, params=params)
    data = response.json()

    if data['ret_code'] == 0:
        return data['result']
    else:
        print("Error:", data['ret_msg'])
        return []

# Save data to CSV
def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

def main():
    symbol = 'BTCUSDT'  # Replace with your trading pair
    interval = 'D'  # Replace with desired interval (1, 5, 15, 30, 60, D, W, M)
    data = get_historical_data(symbol, interval)

    if data:
        save_to_csv(data, f"{symbol}.csv")
        print(f"Data saved to {symbol}.csv")

if __name__ == "__main__":
    main()