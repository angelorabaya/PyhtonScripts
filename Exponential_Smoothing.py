import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def get_EXSMOOTH(currency):
    # Load the dataset
    df = pd.read_csv(currency)

    # Ensure the Date column is in datetime format and sort in ascending order
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # Set the Date as the index
    df.set_index('Date', inplace=True)

    # Use the 'Close' prices for prediction
    close_prices = df['Close']
    last_close_price = close_prices.iloc[-1]

    # Fit the Exponential Smoothing model
    model = ExponentialSmoothing(close_prices, trend='add', seasonal='add', seasonal_periods=30)
    model_fit = model.fit()

    # Forecast the next 30 days
    forecast = model_fit.forecast(steps=30)

    # Create a DataFrame for the results
    forecast_df = pd.DataFrame(forecast, columns=['Predicted_Close'])
    forecast_df.index = pd.date_range(start=close_prices.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')

    # Print the forecast
    #print(forecast_df)
    #print(f"predicted {forecast_df.iloc[-1][-1]} last close {last_close_price}")
    price_predicted = forecast_df.iloc[-1].iloc[-1]

    if price_predicted > last_close_price:
        trend = "Bullish"
    elif price_predicted < last_close_price:
        trend = "Bearish"
    else:
        trend = "Neutral"

    return trend
