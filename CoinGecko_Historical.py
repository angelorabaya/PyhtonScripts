import csv
from pycoingecko import CoinGeckoAPI
from datetime import datetime, timezone

# Initialize CoinGecko API client
cg = CoinGeckoAPI()

# Function to convert Unix timestamp to a readable date
def unix_to_date(timestamp):
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime('%Y-%m-%d')

# Retrieve historical market data
def get_historical_data(coin_id, vs_currency, days):
    data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency=vs_currency, days=days)
    return data

# Save data to CSV
def save_to_csv(data, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Date', 'Price', 'Market Cap', 'Volume'])

        for i in range(len(data['prices'])):
            date = unix_to_date(data['prices'][i][0] / 1000)
            price = data['prices'][i][1]
            market_cap = data['market_caps'][i][1]
            volume = data['total_volumes'][i][1]
            writer.writerow([date, price, market_cap, volume])

# Parameters
coin_id = 'bitcoin'  # Change this to the desired cryptocurrency ID
vs_currency = 'usd'
days = '365'  # Number of days for historical data
filename = 'crypto_data.csv'

# Fetch and save data
historical_data = get_historical_data(coin_id, vs_currency, days)
save_to_csv(historical_data, filename)

print(f"Data saved to {filename}")