import requests

def get_crypto_data(crypto_id):
    url = f'https://api.coingecko.com/api/v3/coins/{crypto_id}'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def analyze_fundamentals(crypto_data):
    if not crypto_data:
        print("Error fetching data")
        return

    name = crypto_data['name']
    symbol = crypto_data['symbol']
    market_cap = crypto_data['market_data']['market_cap']['usd']
    circulating_supply = crypto_data['market_data']['circulating_supply']
    total_supply = crypto_data['market_data']['total_supply']
    max_supply = crypto_data['market_data']['max_supply']
    current_price = crypto_data['market_data']['current_price']['usd']
    volume_24h = crypto_data['market_data']['total_volume']['usd']

    print(f"Name: {name} ({symbol.upper()})")
    print(f"Market Cap: ${market_cap:,.2f}")
    print(f"Current Price: ${current_price:,.2f}")
    print(f"24h Trading Volume: ${volume_24h:,.2f}")
    print(f"Circulating Supply: {circulating_supply:,.0f}")
    print(f"Total Supply: {total_supply:,.0f}")
    print(f"Max Supply: {max_supply if max_supply else 'N/A'}")

if __name__ == "__main__":
    crypto_id = "cardano"  #input("Enter the cryptocurrency ID (e.g., bitcoin, ethereum): ").lower()
    crypto_data = get_crypto_data(crypto_id)
    analyze_fundamentals(crypto_data)