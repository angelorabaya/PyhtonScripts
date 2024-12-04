import requests
from datetime import datetime, timedelta

# Define the URL and API key for CryptoCompare
price_url = 'https://min-api.cryptocompare.com/data/pricemultifull'
historical_url = 'https://min-api.cryptocompare.com/data/v2/histoday'
api_key = 'edcffa8d2e37e4070f4bdcafc78fd8e54ac12745718ccf088e3236aa5523be37'  # Replace with your CryptoCompare API key

# Parameters for the current price request
price_params = {
    'fsyms': 'DOT',  # Polkadot's symbol
    'tsyms': 'USD',  # Currency to compare
    'api_key': api_key
}

# Make the request to CryptoCompare for current data
price_response = requests.get(price_url, params=price_params)
price_data = price_response.json()

# Extract relevant data
market_metrics = price_data['DISPLAY']['DOT']['USD']
raw_data = price_data['RAW']['DOT']['USD']

# Calculate the Market Cap to Volume Ratio
market_cap_to_volume_ratio = float(raw_data['MKTCAP']) / float(raw_data['TOTALVOLUME24HTO'])

# Parameters for the historical data request
historical_params = {
    'fsym': 'DOT',  # Polkadot's symbol
    'tsym': 'USD',  # Currency to compare
    'limit': 30,    # Last 30 days
    'api_key': api_key
}

# Make the request to CryptoCompare for historical data
historical_response = requests.get(historical_url, params=historical_params)
historical_data = historical_response.json()

# Calculate the price change over 7 and 30 days
if historical_data.get('Data') and historical_data['Data'].get('Data'):
    historical_prices = historical_data['Data']['Data']
    price_7_days_ago = historical_prices[-8]['close']
    price_30_days_ago = historical_prices[0]['close']
    current_price = raw_data['PRICE']

    price_change_7d = ((current_price - price_7_days_ago) / price_7_days_ago) * 100
    price_change_30d = ((current_price - price_30_days_ago) / price_30_days_ago) * 100
else:
    price_change_7d = 'N/A'
    price_change_30d = 'N/A'

# Display the information
print("Market Metrics:")
print(f"Current Price (USD): {market_metrics['PRICE']}")
print(f"Market Cap (USD): {market_metrics['MKTCAP']}")
print(f"Trading Volume (24h): {market_metrics['VOLUME24HOURTO']}")
print(f"Market Cap Rank: N/A")  # Rank might need another API call
print(f"Price Change (24h): {market_metrics['CHANGEPCT24HOUR']}%")
print(f"Price Change (7d): {price_change_7d:.2f}%")
print(f"Price Change (30d): {price_change_30d:.2f}%")

print("\nSupply Metrics:")
print(f"Circulating Supply: {raw_data['SUPPLY']}")
print(f"Total Supply: N/A")  # Total supply may need another endpoint
print(f"Supply Ratio: N/A")  # Requires total supply
print(f"Fully Diluted Valuation: N/A")  # Requires total supply

print("\nNetwork Value Metrics:")
print(f"NVT Ratio: N/A")  # Requires additional data
print(f"Market Cap to Volume Ratio: {market_cap_to_volume_ratio:.2f}")