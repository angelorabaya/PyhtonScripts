import pandas as pd
import numpy as np
from typing import List, Tuple

class CandlePatternDetector:
    def __init__(self, df: pd.DataFrame,
                 confirmation_period: int = 3,
                 volume_factor: float = 1.5,
                 trend_periods: int = 14):
        """
        Initialize the detector with configuration parameters

        Args:
            df: DataFrame with OHLCV data
            confirmation_period: Number of candles to confirm trend
            volume_factor: Minimum volume increase factor for confirmation
            trend_periods: Number of periods for trend calculation
        """
        self.df = df.copy()
        self.confirmation_period = confirmation_period
        self.volume_factor = volume_factor
        self.trend_periods = trend_periods
        self._prepare_data()

    def _prepare_data(self):
        """Prepare technical indicators and necessary calculations"""
        # Basic candle calculations
        self.df['Body'] = self.df['Close'] - self.df['Open']
        self.df['Upper_Shadow'] = self.df['High'] - self.df[['Open', 'Close']].max(axis=1)
        self.df['Lower_Shadow'] = self.df[['Open', 'Close']].min(axis=1) - self.df['Low']
        self.df['Body_Size'] = abs(self.df['Body'])

        # Average calculations
        self.df['Avg_Body'] = self.df['Body_Size'].rolling(window=20).mean()
        self.df['Avg_Volume'] = self.df['Volume'].rolling(window=20).mean()

        # Trend indicators
        self.df['SMA'] = self.df['Close'].rolling(window=self.trend_periods).mean()
        self.df['EMA'] = self.df['Close'].ewm(span=self.trend_periods, adjust=False).mean()

        # Momentum indicators
        self.df['RSI'] = self._calculate_rsi(self.df['Close'], periods=14)

        # Volatility
        self.df['ATR'] = self._calculate_atr(periods=14)
        self.df['Avg_ATR'] = self.df['ATR'].rolling(window=20).mean()

    def _calculate_rsi(self, prices: pd.Series, periods: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_atr(self, periods: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = self.df['High'] - self.df['Low']
        high_close = abs(self.df['High'] - self.df['Close'].shift())
        low_close = abs(self.df['Low'] - self.df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=periods).mean()
        return atr

    def _is_trend_reversal_confirmed(self, index: int, pattern_type: str) -> bool:
        """
        Confirm if the reversal pattern is valid based on multiple factors
        """
        if index >= len(self.df) - self.confirmation_period:
            return False

        current = self.df.iloc[index]

        # Volume confirmation
        volume_increased = current['Volume'] > self.volume_factor * current['Avg_Volume']

        # Trend confirmation
        if pattern_type.startswith('Bullish'):
            trend_down = all(self.df.iloc[index + 1:index + self.confirmation_period + 1]['Close'] <
                             self.df.iloc[index + 1:index + self.confirmation_period + 1]['SMA'])
            rsi_oversold = current['RSI'] < 30
            momentum_confirmed = True

        else:  # Bearish patterns
            trend_up = all(self.df.iloc[index + 1:index + self.confirmation_period + 1]['Close'] >
                           self.df.iloc[index + 1:index + self.confirmation_period + 1]['SMA'])
            rsi_overbought = current['RSI'] > 70
            momentum_confirmed = True

        # Volatility confirmation
        volatility_confirmed = current['ATR'] > current['Avg_ATR']

        if pattern_type.startswith('Bullish'):
            return volume_increased and trend_down and rsi_oversold and volatility_confirmed
        else:
            return volume_increased and trend_up and rsi_overbought and volatility_confirmed

    def detect_patterns(self) -> List[Tuple[str, str, float]]:
        """
        Detect and confirm reversal patterns
        Returns: List of tuples containing (Date, Pattern, Reliability Score)
        """
        patterns = []

        for i in range(len(self.df) - 1):
            current = self.df.iloc[i]
            prev = self.df.iloc[i + 1]
            reliability_score = 0
            pattern_found = False
            pattern_type = ""

            # Hammer pattern (bullish)
            if (current['Body'] > 0 and
                    current['Lower_Shadow'] > 2 * abs(current['Body']) and
                    current['Upper_Shadow'] < abs(current['Body']) and
                    prev['Body'] < 0):
                pattern_type = "Hammer (Bullish)"
                pattern_found = True
                reliability_score = 0.6

            # Shooting Star pattern (bearish)
            elif (current['Body'] < 0 and
                  current['Upper_Shadow'] > 2 * abs(current['Body']) and
                  current['Lower_Shadow'] < abs(current['Body']) and
                  prev['Body'] > 0):
                pattern_type = "Shooting Star (Bearish)"
                pattern_found = True
                reliability_score = 0.6

            # Bullish Engulfing
            elif (current['Body'] > 0 and
                  prev['Body'] < 0 and
                  current['Open'] < prev['Close'] and
                  current['Close'] > prev['Open']):
                pattern_type = "Bullish Engulfing"
                pattern_found = True
                reliability_score = 0.8

            # Bearish Engulfing
            elif (current['Body'] < 0 and
                  prev['Body'] > 0 and
                  current['Open'] > prev['Close'] and
                  current['Close'] < prev['Open']):
                pattern_type = "Bearish Engulfing"
                pattern_found = True
                reliability_score = 0.8

            # Doji
            elif abs(current['Body']) < 0.1 * current['Avg_Body']:
                if (current['Upper_Shadow'] > 2 * abs(current['Body']) and
                        current['Lower_Shadow'] > 2 * abs(current['Body'])):
                    pattern_type = "Doji"
                    pattern_found = True
                    reliability_score = 0.5

            if pattern_found:
                # Confirm pattern and adjust reliability score
                if self._is_trend_reversal_confirmed(i, pattern_type):
                    reliability_score += 0.2

                    # Additional reliability factors
                    if current['Volume'] > 2 * current['Avg_Volume']:
                        reliability_score += 0.1
                    if abs(current['Body']) > 1.5 * current['Avg_Body']:
                        reliability_score += 0.1

                    patterns.append((
                        current['Date'],
                        pattern_type,
                        min(round(reliability_score, 2), 1.0)  # Cap at 1.0
                    ))

        return patterns


def main():
    try:
        # Read the CSV file
        df = pd.read_csv('BTCUSDT.csv')

        # Initialize detector
        detector = CandlePatternDetector(
            df,
            confirmation_period=3,
            volume_factor=1.5,
            trend_periods=14
        )

        # Detect patterns
        patterns = detector.detect_patterns()

        # Print results
        if patterns:
            print("\nDetected Reversal Patterns:")
            print("Date | Pattern | Reliability Score")
            print("-" * 50)
            for date, pattern, score in patterns:
                print(f"{date} | {pattern} | {score:.2f}")
        else:
            print("No reversal patterns detected")

    except FileNotFoundError:
        print("Error: CSV file not found")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()