import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from AutoARIMA import get_Arima_Values

def get_ARIMA(currency):
    # Load data
    file_path = currency
    data = pd.read_csv(file_path)

    # Parse dates and set index with frequency
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data = data.asfreq('D')  # Set frequency to daily

    # Use the 'Close' prices for ARIMA
    close_prices = data['Close']

    p,d,q = get_Arima_Values(file_path)

    # Fit the ARIMA model (adjust order parameters if necessary)
    model = ARIMA(close_prices, order=(p, d, q), enforce_stationarity=False)  # ARIMA(p, d, q) parameters
    model_fit = model.fit()

    # Make predictions
    forecast = model_fit.forecast(steps=1)
    #predicted_price = forecast[0]
    predicted_price = forecast.iloc[0]
    last_price = close_prices.iloc[-1]

    # Determine if the market is bullish or bearish
    if predicted_price > last_price:
        market_trend = 'Bullish'
    else:
        market_trend = 'Bearish'

    # Print the results
    #print(f"Last Closing Price: {last_price:.2f}")
    #print(f"Predicted Price: {predicted_price:.2f}")
    #print(f"Market Trend: {market_trend}")

    return [market_trend,p,d,q]