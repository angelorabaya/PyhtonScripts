import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load the dataset
file_path = 'BNBUSDT.csv'  # Change this to your file path
data = pd.read_csv(file_path)

# Parse the date column
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Select 'Close' prices for prediction
series = data['Close']

# Split the data into train/test sets
train_size = int(len(series) * 0.8)
train, test = series[:train_size], series[train_size:]

# Define the SARIMA model
p, d, q = 5, 1, 4  # Non-seasonal orders
P, D, Q, s = 5, 1, 4, 12  # Seasonal orders

model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, s))
model_fit = model.fit()

# Forecasting future prices
n_periods = len(test)
forecast = model_fit.forecast(steps=n_periods)

# Combine train, test, and forecast for plotting
forecast_index = test.index
combined = pd.Series(np.concatenate([train, forecast]), index=train.index.append(forecast_index))

# Count bullish and bearish predictions
last_observed = train.iloc[-1]
bullish_count = (forecast > last_observed).sum()
bearish_count = (forecast < last_observed).sum()

# Print future predictions and counts
print("Future Predictions:")
print(forecast)
print(f"\nBullish Predictions Count: {bullish_count}")
print(f"Bearing Predictions Count: {bearish_count}")