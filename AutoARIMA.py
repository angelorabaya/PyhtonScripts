import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima

# Load the data
data = pd.read_csv('EURUSD.csv', parse_dates=['Date'], index_col='Date')

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