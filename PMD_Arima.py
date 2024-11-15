import pandas as pd
import numpy as np
from pmdarima import auto_arima
from datetime import timedelta

# Read the CSV file
def load_and_prepare_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Sort the dataframe by Date in ascending order
    df = df.sort_values('Date', ascending=True)

    # Set Date as index
    df = df.set_index('Date')
    return df

# Train ARIMA model and make predictions
def train_and_predict(data, forecast_periods=30):
    # Fit auto ARIMA model
    model = auto_arima(data['Close'],
                       start_p=1, start_q=1,
                       max_p=3, max_q=3,
                       m=1,
                       start_P=0, seasonal=False,
                       d=1, D=1, trace=True,
                       error_action='ignore',
                       suppress_warnings=True,
                       stepwise=True)

    best_order = model.order
    p, d, q = best_order

    # Generate future dates (excluding weekends)
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1),
                                 periods=forecast_periods,
                                 freq='B')

    # Make predictions
    forecast, conf_int = model.predict(n_periods=forecast_periods,
                                       return_conf_int=True)

    return forecast, conf_int, future_dates, p, d, q

def get_PDARIMA(currency):
    # File path
    file_path = currency

    # Number of days to forecast
    forecast_periods = 30

    try:
        # Load and prepare data
        data = load_and_prepare_data(file_path)

        # Print the date range of historical data
        #print(f"\nHistorical data range:")
        #print(f"Start date: {data.index[0]}")
        #print(f"End date: {data.index[-1]}")

        # Train model and get predictions
        forecast, conf_int, future_dates, p, d, q = train_and_predict(data, forecast_periods)

        # Create and display prediction dataframe
        prediction_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Price': np.round(forecast, 2),
            'CI_Lower': np.round(conf_int[:, 0], 2),
            'CI_Upper': np.round(conf_int[:, 1], 2)
        })

        #print("\nPredicted Prices (Next 30 Business Days):")
        #print(prediction_df.to_string())

        # Compare last close price and first predicted price
        close_prices = data['Close']
        last_close_price = close_prices.iloc[-1]
        price_predicted = prediction_df['Predicted_Price']
        last_predicted_price = price_predicted.iloc[-1]

        if last_predicted_price > last_close_price:
            trend = "Bullish"
        elif last_predicted_price < last_close_price:
            trend = "Bearish"
        else:
            trend = "Neutral"

        return [trend, p, d, q]

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    get_PDARIMA("BTCUSDT.csv")