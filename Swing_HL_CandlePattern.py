import pandas as pd


def is_bullish_reversal(candle):
    return candle['Close'] > candle['Open'] and (candle['High'] - candle['Close']) <= (
                candle['Close'] - candle['Open']) * 2


def is_bearish_reversal(candle):
    return candle['Close'] < candle['Open'] and (candle['Close'] - candle['Low']) <= (
                candle['Open'] - candle['Close']) * 2


def detect_swing_high_low(data):
    signals = []

    for i in range(1, len(data) - 1):
        current = data.iloc[i]
        previous = data.iloc[i - 1]
        next_candle = data.iloc[i + 1]

        # Detect Swing High
        if current['High'] > previous['High'] and current['High'] > next_candle['High']:
            if is_bearish_reversal(current):
                signals.append((current['Date'], 'Swing High with Bearish Reversal'))

        # Detect Swing Low
        elif current['Low'] < previous['Low'] and current['Low'] < next_candle['Low']:
            if is_bullish_reversal(current):
                signals.append((current['Date'], 'Swing Low with Bullish Reversal'))

    return signals


def main():
    # Load data from CSV
    file_path = 'BTCUSDT.csv'  # Update this to your CSV file path
    data = pd.read_csv(file_path)

    # Detect swing highs and lows
    signals = detect_swing_high_low(data)

    # Print results
    if signals:
        for date, pattern in signals:
            print(f"Date: {date}, Pattern: {pattern}")
    else:
        print("No swing high/low with reversal patterns detected.")


if __name__ == "__main__":
    main()