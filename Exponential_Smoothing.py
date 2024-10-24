import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load historical data from CSV
def load_data(file_path):
    # Load the CSV file and set the Date column as the index
    data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

    # Sort the DataFrame by the index (Date) in descending order
    data = data.sort_index(ascending=True)

    # Return the sorted DataFrame
    return data

# Forecast future prices using ETS
def forecast_prices(data, steps=7):
    model = ExponentialSmoothing(data['Close'], trend='add', seasonal='add', seasonal_periods=12)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps)
    return forecast

# Main function to run the script
def main():
    file_path = 'DJT.csv'  # Replace with your CSV file path
    historical_data = load_data(file_path)

    # Ensure there are enough data points
    if len(historical_data) < 12:
        print("Not enough data points for forecasting. Requires at least 12.")
        return

    # Generate future prices
    future_prices = forecast_prices(historical_data)

    # Print future prices
    print("Future prices for the next 7 days:")
    for i, price in enumerate(future_prices, start=1):
        print(f"Day {i}: {price:.2f}")


if __name__ == "__main__":
    main()