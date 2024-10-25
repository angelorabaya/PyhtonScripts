import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Load data
file_path = 'BTCUSDT.csv'  # Replace with your CSV file path
data = pd.read_csv(file_path)

# Parse dates and set index with frequency
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data = data.asfreq('D')  # Set frequency to daily

# Use the 'Close' prices for ARIMA
close_prices = data['Close']

# Fit the ARIMA model (adjust order parameters if necessary)
model = ARIMA(close_prices, order=(1, 1, 0))  # ARIMA(p, d, q) parameters
model_fit = model.fit()

# Make predictions for the next week (7 days)
forecast = model_fit.forecast(steps=7)
predicted_prices = forecast.values  # Get the predicted prices
last_price = close_prices.iloc[-1]

# Determine market trend for each day in the forecast
trends = []
for predicted_price in predicted_prices:
    if predicted_price > last_price:
        trends.append('Bullish')
    else:
        trends.append('Bearish')

# Print results
print(f"Last Closing Price: {last_price:.2f}")
for day, predicted_price, trend in zip(range(1, 8), predicted_prices, trends):
    print(f"Predicted Price for Day {day}: {predicted_price:.2f}, Candle is {trend}")