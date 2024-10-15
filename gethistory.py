import yfinance as yf
import pandas as pd


def get_currency_data(currency_pair, start_date, end_date):
    """
    Fetch historical currency data from Yahoo Finance.

    Parameters:
    - currency_pair (str): The currency pair to fetch (e.g., 'EURUSD=X').
    - start_date (str): Start date for the historical data (format: 'YYYY-MM-DD').
    - end_date (str): End date for the historical data (format: 'YYYY-MM-DD').

    Returns:
    - DataFrame: Historical data as a pandas DataFrame.
    """
    # Fetch data
    data = yf.download(currency_pair, start=start_date, end=end_date)

    return data


if __name__ == "__main__":
    # Define currency pair, start date and end date
    currency_pair = 'AAPL'  # Euro to US Dollar
    start_date = '2018-01-01'  # Start date
    end_date = '2024-10-11'  # End date

    # Get the historical data
    historical_data = get_currency_data(currency_pair, start_date, end_date)

    # Display the historical data
    print(historical_data)

    # Optionally, save to CSV
    historical_data.to_csv(f'{currency_pair}_{start_date}_to_{end_date}.csv')