import requests

# Replace 'YOUR_API_KEY' with your actual Alpha Vantage API key
API_KEY = '6XJOAVQKQGMDSWC8'
BASE_URL = 'https://www.alphavantage.co/query'

def get_sentiment(symbol):
    # Make a request to the Alpha Vantage API for crypto data
    parameters = {
        'function': 'DIGITAL_CURRENCY_DAILY',
        'symbol': symbol,
        'market': 'USD',
        'apikey': API_KEY
    }

    response = requests.get(BASE_URL, params=parameters)
    data = response.json()

    # Print the response for debugging
    print("API Response:", data)

    # Check if the expected key is in the response
    if 'Time Series (Digital Currency Daily)' in data:
        timeseries = data['Time Series (Digital Currency Daily)']
        if timeseries:  # Ensure timeseries isn't empty
            # Extract the latest day's data for sentiment analysis
            latest_date = sorted(timeseries.keys())[0]
            latest_data = timeseries[latest_date]

            close_price = float(latest_data['4a. close (USD)'])
            open_price = float(latest_data['1a. open (USD)'])

            # Calculate sentiment
            sentiment = 'Bullish' if close_price > open_price else 'Bearish'
            return {
                'date': latest_date,
                'close_price': close_price,
                'open_price': open_price,
                'sentiment': sentiment
            }
    else:
        return {"error": data.get("Error Message", "Unable to fetch data")}

if __name__ == "__main__":
    symbol = 'BTC'
    sentiment_data = get_sentiment(symbol)
    print(sentiment_data)