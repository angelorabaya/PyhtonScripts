import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller

# Load the data
df = pd.read_csv('d:/BTCUSDT.csv', parse_dates=['Date'], index_col='Date')
df = df[['Close']]  # Focus on the 'Close' price

# Check for stationarity
def check_stationarity(series):
    result = adfuller(series)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'{key}: {value}')


check_stationarity(df['Close'])

# Plot the closing prices
plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='BTC Closing Prices')
plt.title('BTC/USD Closing Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# ARIMA model
# Here p,d,q values need to be selected properly, you can use model selection criteria like AIC/BIC
model_arima = ARIMA(df['Close'], order=(1, 1, 0))  # Example order
model_arima_fit = model_arima.fit()
print(model_arima_fit.summary())

# Forecasting with ARIMA
arima_forecast = model_arima_fit.forecast(steps=30)  # Predict next 30 days
plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='BTC Closing Prices')
plt.plot(pd.date_range(start=df.index[-1], periods=30, freq='D'), arima_forecast, label='ARIMA Forecast', color='red')
plt.title('ARIMA Forecast')
plt.legend()
plt.show()

# Exponential smoothing
model_exp_smooth = ExponentialSmoothing(df['Close'], trend='add', seasonal='add', seasonal_periods=12).fit()
exp_smooth_forecast = model_exp_smooth.forecast(steps=30)

# Plot Exponential Smoothing
plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='BTC Closing Prices')
plt.plot(pd.date_range(start=df.index[-1], periods=30, freq='D'), exp_smooth_forecast,
         label='Exponential Smoothing Forecast', color='green')
plt.title('Exponential Smoothing Forecast')
plt.legend()
plt.show()