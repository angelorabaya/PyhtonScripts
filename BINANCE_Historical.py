from calendar import month

import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time


def fetch_binance_historical_data(symbol, timeframe='1h', start_date=None, end_date=None):
    """
    Fetch historical cryptocurrency data from Binance API

    Parameters:
    - symbol: Trading pair (e.g., 'BTC/USDT')
    - timeframe: Candle interval ('1m', '5m', '15m', '1h', '4h', '1d')
    - start_date: Start date for data retrieval (optional)
    - end_date: End date for data retrieval (optional)

    Returns: DataFrame with historical price data
    """
    try:
        # Initialize Binance exchange
        exchange = ccxt.binance({
            'enableRateLimit': True,  # Enable rate limiting to avoid API restrictions
            'options': {
                'defaultType': 'future'  # Use futures market, change to 'spot' if needed
            }
        })

        # Set default date range if not provided
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()

        # Convert dates to milliseconds
        start_timestamp = int(start_date.timestamp() * 1000)
        end_timestamp = int(end_date.timestamp() * 1000)

        # Fetch historical data
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=start_timestamp, limit=1000)

        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Date'] = pd.to_datetime(df['Date'], unit='ms')

        return df

    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


def save_to_csv(df, filename=None):
    """
    Save DataFrame to CSV file

    Parameters:
    - df: DataFrame to save
    - filename: Custom filename (optional)
    """
    if df is not None:
        if filename is None:
            filename = f"{symbol.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")


def main():
    # Configuration
    symbols = ['BTC/USDT']
    timeframes = ['1d']

    # Date range (optional)
    start_date = datetime.now() - timedelta(days=999)
    end_date = datetime.now()

    for symbol in symbols:
        for timeframe in timeframes:
            print(f"Fetching data for {symbol} - {timeframe} timeframe")

            df = fetch_binance_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )

            if df is not None:
                filename = f"{symbol.replace('/', '_')}_{timeframe}.csv"
                save_to_csv(df, filename)

            # Respect API rate limits
            time.sleep(1)


if __name__ == "__main__":
    main()