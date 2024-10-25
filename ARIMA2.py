import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

def automatic_differencing(series):
    diff_series = series
    d = 0  # Differencing order
    while True:
        result = adfuller(diff_series)
        if result[1] < 0.05:  # If p-value is less than 0.05, it's stationary
            break
        diff_series = diff_series.diff().dropna()  # Perform differencing
        d += 1  # Increment differencing order
    return diff_series.dropna(), d  # Drop NaN values before returning

# Load data
file_path = 'DOGEUSDT.csv'
data = pd.read_csv(file_path)

# Parse dates and set index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data = data.asfreq('D')

# Use the 'Close' prices for differencing
close_prices = data['Close']

# Check and handle NaN and infinite values
close_prices.replace([float('inf'), float('-inf')], float('nan'), inplace=True)
close_prices.dropna(inplace=True)

# Automatically difference the series until stationary
max_diff, d_order = automatic_differencing(close_prices)

# Fit ARIMA model with the differenced series
model = ARIMA(max_diff, order=(0, d_order, 5))
try:
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=1)
    predicted_price = forecast.iloc[0]
    last_price = close_prices.iloc[-1]

    market_trend = 'Bullish' if predicted_price > last_price else 'Bearish'

    print(f"Last Closing Price: {last_price:.2f}")
    print(f"Predicted Price: {predicted_price:.2f}")
    print(f"Market Trend: {market_trend}")

except Exception as e:
    print(f"An error occurred: {str(e)}")