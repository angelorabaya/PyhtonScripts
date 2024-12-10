import requests
import pandas as pd
from datetime import datetime

# Configuration
API_KEY = 'edcffa8d2e37e4070f4bdcafc78fd8e54ac12745718ccf088e3236aa5523be37'  # Replace with your CryptoCompare API key
CRYPTOCURRENCY = 'SHIB'      # Cryptocurrency symbol (e.g., 'BTC' for Bitcoin)
CURRENCY = 'USDT'            # Desired quote currency (e.g., 'USD')
HISTORICAL_DATA_URL = 'https://min-api.cryptocompare.com/data/v2/histoday'
LIMIT = 2000                # Maximum number of data points to retrieve
TO_TS = int(datetime.now().timestamp())  # Current timestamp

# Function to retrieve historical data
def fetch_historical_data():
    params = {
        'fsym': CRYPTOCURRENCY,
        'tsym': CURRENCY,
        'limit': LIMIT,
        'toTs': TO_TS,
        'api_key': API_KEY
    }

    response = requests.get(HISTORICAL_DATA_URL, params=params)
    
    if response.status_code == 200:
        data = response.json()
        return data['Data']['Data']
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

# Function to save data to CSV
def save_to_csv(data):
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['time'], unit='s')  # Convert Unix timestamp to datetime
    df.to_csv(f'{CRYPTOCURRENCY}_{CURRENCY}.csv', index=False)
    print(f"Data saved to {CRYPTOCURRENCY}_{CURRENCY}.csv")

# Main execution
if __name__ == "__main__":
    historical_data = fetch_historical_data()
    if historical_data:
        save_to_csv(historical_data)