import pandas as pd
from arch import arch_model

# Load data from CSV
data = pd.read_csv('BTCUSDT.csv')

# Ensure 'Date' is parsed as datetime
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Use 'Close' price for the GARCH model
returns = data['Close'].pct_change().dropna()

# Rescale returns
scaling_factor = 10
rescaled_returns = returns * scaling_factor

# Fit GARCH model
model = arch_model(rescaled_returns, vol='Garch', p=1, q=1)
model_fit = model.fit(disp='off')

# Forecast future returns (next 7 periods)
n_forecast = 10  # Change to 7 days
forecast = model_fit.forecast(horizon=n_forecast)

# Output forecast results
mean_returns_forecast = forecast.mean.values[-1, :]

# Determine bullish or bearish sentiment based on the first forecasted return
if mean_returns_forecast[0] > 0:
    sentiment = 'Bullish'
else:
    sentiment = 'Bearish'

# Display the sentiment
print(f'The forecast is: {sentiment}')