import requests
import pandas as pd

# Constants
API_KEY = '2YKB3WHR8DQW5JAN'  # replace with your Alpha Vantage API key 6XJOAVQKQGMDSWC8 Z796Q911XLQXDL3P N13H2YJPT5KATOL2
SYMBOL = 'NEARUSDT'  # replace with your desired stock symbol
TIME_FRAME = 'daily'  # available options: daily, weekly, monthly
OUTPUT_SIZE = 'full'  # 'compact' or 'full'

crypto_list_all = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "DOTUSDT", "LINKUSDT", "MATICUSDT",
    "AVAXUSDT", "UNIUSDT", "ARBUSDT", "AAVEUSDT", "MKRUSDT",
    "COMPUSDT", "NEARUSDT", "ATOMUSDT", "ALGOUSDT", "XTZUSDT",
    "OPUSDT", "LRCUSDT", "IMXUSDT", "ERAUSDT",
    "AXSUSDT", "MANAUSDT", "SANDUSDT", "GALAUSDT", "IMXUSDT",
    "GRTUSDT", "BATUSDT", "FILUSDT", "ICPUSDT", "HBARUSDT",
    "QNTUSDT", "VETUSDT", "XMRUSDT", "ZECUSDT", "SCRTUSDT",
    "FTTUSDT", "CROUSDT", "OKBUSDT", "KCSUSDT", "SHIBUSDT",
    "PEPEUSDT", "FLOKIUSDT", "ATOMUSDT", "KSMUSDT", "ONEUSDT",
    "RUNEUSDT", "WBTCUSDT", "RENBTCUSDT", "FETUSDT", "AGIXUSDT",
    "OCEANUSDT"
]

crypto_list = ["BTCUSDT"]

# Functions
def fetch_historical_data(symbol, api_key, time_frame, output_size):
    base_url = 'https://www.alphavantage.co/query'
    params = {
        'function': f'TIME_SERIES_{time_frame.upper()}',
        'symbol': symbol,
        'apikey': api_key,
        'outputsize': output_size,
        'datatype': 'json'
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error fetching data: {response.status_code} {response.text}")

def convert_to_dataframe(data, time_frame):
    time_series_key = f'Time Series ({time_frame.capitalize()})'
    if time_series_key in data:
        df = pd.DataFrame(data[time_series_key]).T
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df.index.name = 'Date'
        df = df.astype(float)
        df.index = pd.to_datetime(df.index)
        return df
    else:
        raise Exception("Invalid data format from API response")

def save_to_csv(df, filename):
    df.to_csv(filename)
    print(f"Data saved to {filename}")

# Main execution
if __name__ == '__main__':
    try:
        #historical_data = fetch_historical_data(SYMBOL, API_KEY, TIME_FRAME, OUTPUT_SIZE)
        #df = convert_to_dataframe(historical_data, TIME_FRAME)
        #save_to_csv(df, f'{SYMBOL}.csv')

        for crypto in crypto_list:
            historical_data = fetch_historical_data(crypto, API_KEY, TIME_FRAME, OUTPUT_SIZE)
            df = convert_to_dataframe(historical_data, TIME_FRAME)
            save_to_csv(df, f'{crypto}.csv')

    except Exception as e:
        print(str(e))