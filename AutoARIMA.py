import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima

# Load the data
data = pd.read_csv('BTCUSDT.csv', parse_dates=['Date'], index_col='Date')

# Ensure the 'Close' column is in numeric format
data['Close'] = pd.to_numeric(data['Close'], errors='coerce')

# Resample the data to daily frequency (if needed)
data = data['Close'].resample('D').mean()

# Drop any missing values
data = data.dropna()

# Fit auto-ARIMA model
model = auto_arima(data, seasonal=False, stepwise=True, trace=True, suppress_warnings=True)

# Display model summary
print(model.summary())

# Forecast future values
#n_periods = 30  # Number of days to forecast
#forecast, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)

# Create a time index for forecasted values
#forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=n_periods)

# Create a DataFrame for forecasted values
#forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=['Forecast'])

# Plot original data and forecast
#plt.figure(figsize=(12, 6))
#plt.plot(data, label='Historical Data', color='blue')
#plt.plot(forecast_df, label='Forecast', color='red')
#plt.fill_between(forecast_index, conf_int[:, 0], conf_int[:, 1], color='pink', alpha=0.3, label='Confidence Interval')
#plt.title('Price Forecast')
#plt.xlabel('Date')
#plt.ylabel('Price (USD)')
#plt.legend()
#plt.show()