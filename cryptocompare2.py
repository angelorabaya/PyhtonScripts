import requests
import pandas as pd

def get_crypto_data(symbol, comparison_symbol, limit, exchange=''):
    url = 'https://min-api.cryptocompare.com/data/v2/histoday'

    api_key = 'edcffa8d2e37e4070f4bdcafc78fd8e54ac12745718ccf088e3236aa5523be37'

    parameters = {
        'fsym': symbol,
        'tsym': comparison_symbol,
        'limit': limit,
        'e': exchange,
        'api_key': api_key
    }

    response = requests.get(url, params=parameters)
    data = response.json()

    df = pd.DataFrame(data['Data']['Data'])

    # Convert timestamp to datetime
    df['Date'] = pd.to_datetime(df['time'], unit='s')

    # Select and rename columns
    df = df[['Date', 'open', 'high', 'low', 'close', 'volumefrom']]
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

    # Format numbers to display full decimal values
    pd.set_option('display.float_format', lambda x: '%.8f' % x)

    # Convert numeric columns to float with 8 decimal places
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_columns:
        df[col] = df[col].astype(float).round(8)

    # Sort by date
    df = df.sort_values('Date')

    return df

# Example usage
symbol = 'ADA'  # Cryptocurrency (e.g., BTC, ETH)
comparison = 'USDT'  # Base currency (e.g., USD, EUR)
days = 2000  # Number of days of historical data
exchange = 'Kraken'  # Exchange (CCCAGG is the default aggregated data) Binance,Coinbase,Kraken,Bitfinex,Huobi,Bittrex,Gemini,OKEx,KuCoin,Bitstamp

df = get_crypto_data(symbol, comparison, days, exchange)

# Save to CSV with full decimal places
df.to_csv(f'{symbol}_{comparison}.csv', index=False, float_format='%.4f')

# Display the data with full decimal places
#print(df.head())
print("file created!")