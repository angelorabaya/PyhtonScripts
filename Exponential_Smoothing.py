import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load the CSV data
file_path = 'BTCUSDT.csv'  # Update this line with your file path
data = pd.read_csv(file_path, parse_dates=['Date']).sort_values('Date', ascending=False)

# Check for missing values and handle them (e.g., forward fill)
data['Close'] = data['Close'].ffill()  # Updated line to use ffill() method

# Fit the Exponential Smoothing model using 'Close' price
try:
    model = ExponentialSmoothing(data['Close'], trend='add', seasonal=None).fit()

    # Forecast future prices
    forecast_periods = 10
    forecast = model.forecast(steps=forecast_periods)

    # Output the forecasted prices
    print(f"Forecasted Future Prices for the next {forecast_periods} periods:")
    print('\n'.join([f"Period {i + 1}: {price:.2f}" for i, price in enumerate(forecast)]))

except Exception as e:
    print(f"An error occurred: {e}")