import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time


def get_crypto_data(symbol, timeframe, limit):
    # Initialize Binance exchange (using public API)
    exchange = ccxt.binance()

    try:
        # Fetch OHLCV data
        ohlcv = exchange.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit
        )

        # Convert to DataFrame
        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
        )

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        return df

    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


def calculate_indicators(df):
    results = {}

    # RSI (14 periods)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss
    results['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    results['MACD'] = macd
    results['MACD_Signal'] = signal
    results['MACD_Histogram'] = macd - signal

    # Moving Averages
    results['MA20'] = df['Close'].rolling(window=20).mean()
    results['MA50'] = df['Close'].rolling(window=50).mean()
    results['MA200'] = df['Close'].rolling(window=200).mean()

    # Bollinger Bands
    results['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    results['BB_Upper'] = results['BB_Middle'] + (bb_std * 2)
    results['BB_Lower'] = results['BB_Middle'] - (bb_std * 2)

    # Support and Resistance
    results['PP'] = (df['High'] + df['Low'] + df['Close']) / 3
    results['R1'] = 2 * results['PP'] - df['Low']
    results['S1'] = 2 * results['PP'] - df['High']
    results['R2'] = results['PP'] + (df['High'] - df['Low'])
    results['S2'] = results['PP'] - (df['High'] - df['Low'])

    return results


def analyze_signals(indicators, current_price):
    signals = {
        'RSI': None,
        'MACD': None,
        'BB': None,
        'MA': None,
        'Support_Resistance': None
    }

    # RSI Analysis
    last_rsi = indicators['RSI'].iloc[-1]
    if last_rsi > 70:
        signals['RSI'] = 'Overbought'
    elif last_rsi < 30:
        signals['RSI'] = 'Oversold'
    else:
        signals['RSI'] = 'Neutral'

    # MACD Analysis
    if indicators['MACD'].iloc[-1] > indicators['MACD_Signal'].iloc[-1]:
        signals['MACD'] = 'Bullish'
    else:
        signals['MACD'] = 'Bearish'

    # Bollinger Bands Analysis
    if current_price > indicators['BB_Upper'].iloc[-1]:
        signals['BB'] = 'Overbought'
    elif current_price < indicators['BB_Lower'].iloc[-1]:
        signals['BB'] = 'Oversold'
    else:
        signals['BB'] = 'Within Bands'

    # Moving Average Analysis
    if current_price > indicators['MA20'].iloc[-1] and current_price > indicators['MA50'].iloc[-1]:
        signals['MA'] = 'Bullish'
    elif current_price < indicators['MA20'].iloc[-1] and current_price < indicators['MA50'].iloc[-1]:
        signals['MA'] = 'Bearish'
    else:
        signals['MA'] = 'Mixed'

    # Support/Resistance Analysis
    if current_price > indicators['R1'].iloc[-1]:
        signals['Support_Resistance'] = 'Above R1'
    elif current_price < indicators['S1'].iloc[-1]:
        signals['Support_Resistance'] = 'Below S1'
    else:
        signals['Support_Resistance'] = 'Between S1-R1'

    return signals


def main():
    # Define parameters
    symbol = 'TRX/USDT'
    timeframes = {
        '1h': 500,  # 1 hour data
        '4h': 500,  # 4 hour data
        '1d': 200  # daily data
    }

    analysis_results = {}

    for timeframe, limit in timeframes.items():
        # Get data
        df = get_crypto_data(symbol, timeframe, limit)
        if df is not None:
            # Calculate indicators
            indicators = calculate_indicators(df)

            # Analyze current signals
            current_price = df['Close'].iloc[-1]
            signals = analyze_signals(indicators, current_price)

            analysis_results[timeframe] = {
                'current_price': current_price,
                'signals': signals,
                'last_update': df.index[-1]
            }

    return analysis_results


if __name__ == "__main__":
    results = main()

    # Print results
    for timeframe, data in results.items():
        print(f"\nTimeframe: {timeframe}")
        print(f"Current Price: {data['current_price']:.6f}")
        print(f"Last Update: {data['last_update']}")
        print("\nSignals:")
        for indicator, signal in data['signals'].items():
            print(f"{indicator}: {signal}")
        print("-" * 50)