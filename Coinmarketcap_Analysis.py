import requests

CRYPTO_SYMBOL = "LTC"

def get_crypto_data():
  try:
    # Using CoinMarketCap API (free plan)
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
    parameters = {
      'symbol': CRYPTO_SYMBOL,
      'convert': 'USD'
    }
    headers = {
      'Accepts': 'application/json',
      'X-CMC_PRO_API_KEY': '6d7f0f3d-f7d5-48eb-8d09-d48c3d1eb7a1'  # Replace with your CoinMarketCap API key
    }

    response = requests.get(url, headers=headers, params=parameters)
    response.raise_for_status()  # Raise an exception for bad status codes
    data = response.json()

    # Extracting relevant data
    dot_data = data['data'][CRYPTO_SYMBOL]
    market_data = {
        'Current Price (USD)': dot_data['quote']['USD']['price'],
        'Market Cap (USD)': dot_data['quote']['USD']['market_cap'],
        'Trading Volume (24h)': dot_data['quote']['USD']['volume_24h'],
        'Market Cap Rank': dot_data['cmc_rank'],
        'Price Change (24h)': dot_data['quote']['USD']['percent_change_24h'],
        'Price Change (7d)': dot_data['quote']['USD']['percent_change_7d'],
        'Price Change (30d)': dot_data['quote']['USD']['percent_change_30d']
    }
    supply_data = {
        'Circulating Supply': dot_data['circulating_supply'],
        'Total Supply': dot_data['total_supply'],
        'Supply Ratio': f"{dot_data['circulating_supply'] / dot_data['total_supply'] * 100:.2f}%",
        'Fully Diluted Valuation': dot_data['quote']['USD']['fully_diluted_market_cap']
    }

    # Calculating NVT Ratio (requires additional data or assumptions)
    # Here, we're using a simplified calculation with Market Cap and Volume
    nvt_ratio = market_data['Market Cap (USD)'] / market_data['Trading Volume (24h)']
    network_data = {
        'NVT Ratio': f"{nvt_ratio:.2f}",
        'Market Cap to Volume Ratio': f"{nvt_ratio:.2f}"
    }

    return {
        'Market Metrics': market_data,
        'Supply Metrics': supply_data,
        'Network Value Metrics': network_data
    }

  except requests.exceptions.RequestException as e:
    print(f"Error fetching data: {e}")
    return None

if __name__ == "__main__":
  polkadot_data = get_crypto_data()
  if polkadot_data:
    print("CRYPTOCURRENCY ANALYSIS")
    print(f"Asset: {CRYPTO_SYMBOL}USD")
    print(polkadot_data)