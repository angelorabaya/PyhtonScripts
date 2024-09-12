import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load your currency data
data = pd.read_csv('d:/BTCUSD.csv', parse_dates=['Date'], index_col='Date')

# Set the frequency of the date index (if your data is daily)
data = data.asfreq('D')

# Fit an ARIMA model
model = ARIMA(data['Close'], order=(1, 1, 0))
model_fit = model.fit()

# Forecast future values
forecast = model_fit.forecast(steps=30)

# Plot results
plt.plot(data['Close'], label='Historical Data')
plt.plot(forecast.index, forecast, label='Forecast', color='red')
plt.title('XRP/USD Price Forecast')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
