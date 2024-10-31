import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load and preprocess data
data = pd.read_csv('BTCUSDT.csv', parse_dates=['Date'], index_col='Date')

# Sort and set frequency (daily in this case)
data = data['Close'].sort_index(ascending=False).asfreq('D')

# Split the data into training and test sets
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# Fit the SARIMA model (adjust p, d, q, P, D, Q, s as needed)
model = SARIMAX(train, order=(1, 1, 0), seasonal_order=(1, 1, 0, 12)).fit(disp=False)

# Make predictions
forecast = model.forecast(steps=len(test))

# Evaluate trend direction
trend_direction = ['Bullish' if forecast[i] > forecast[i - 1] else 'Bearish' for i in range(1, len(forecast))]

# Count bullish and bearish trends
bullish_count = trend_direction.count('Bullish')
bearish_count = trend_direction.count('Bearish')

# Prepare final output results
forecast_results = pd.DataFrame({
    'Forecasted Value': forecast,
    'Trend Direction': [''] + trend_direction  # Adding empty string for the first value
})

# Output forecasted values along with trend direction
print("Forecasted values with trends:\n", forecast_results)
print(f"\nNumber of Bullish trends: {bullish_count}")
print(f"Number of Bearish trends: {bearish_count}")