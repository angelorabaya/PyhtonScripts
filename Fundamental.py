import requests

def fetch_crypto_info(symbol):
    # Define the API endpoint
    url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"

    # Make a GET request to the API
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()

        # Extract fundamental information
        if data:
            info = {
                "symbol": data['symbol'],
                "priceChange": data['priceChange'],
                "priceChangePercent": data['priceChangePercent'],
                "weightedAvgPrice": data['weightedAvgPrice'],
                "lastPrice": data['lastPrice'],
                "lastQty": data['lastQty'],
                "bidPrice": data['bidPrice'],
                "askPrice": data['askPrice'],
                "openPrice": data['openPrice'],
                "highPrice": data['highPrice'],
                "lowPrice": data['lowPrice'],
                "volume": data['volume'],
                "quoteVolume": data['quoteVolume'],
                "openTime": data['openTime'],
                "closeTime": data['closeTime'],
                "count": data['count']
            }
            return info
        else:
            return "No data found for the provided symbol."
    else:
        return f"Error: Unable to fetch data (Status code: {response.status_code})."

if __name__ == "__main__":
    # Define the cryptocurrency pair you want to fetch
    symbol = "BNBUSDT"
    crypto_info = fetch_crypto_info(symbol)

    # Print the fetched information
    print(crypto_info)