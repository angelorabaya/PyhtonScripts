import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from sklearn.cluster import DBSCAN


class PriceLevelAnalyzer:
    def __init__(self, file_path,
                 window=20,
                 touch_threshold=0.002,
                 volume_factor=1.5,
                 min_level_distance=0.01):
        """
        Initialize the analyzer with configuration parameters

        Parameters:
        - window: Period for detecting swing points
        - touch_threshold: Percentage threshold for price touching a level
        - volume_factor: Minimum volume multiplier for significant levels
        - min_level_distance: Minimum distance between levels as percentage
        """
        self.window = window
        self.touch_threshold = touch_threshold
        self.volume_factor = volume_factor
        self.min_level_distance = min_level_distance
        self.df = self._load_and_prepare_data(file_path)

    def _load_and_prepare_data(self, file_path):
        """Load and prepare the price data"""
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')

        # Calculate additional technical indicators
        df['HL_avg'] = (df['High'] + df['Low']) / 2
        df['HLC_avg'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['OHLC_avg'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4

        # Calculate volume moving average
        df['Volume_MA'] = df['Volume'].rolling(window=self.window).mean()

        return df

    def _find_swing_points(self):
        """Identify swing high and low points with volume confirmation"""
        # Find local maxima and minima
        high_idx = argrelextrema(self.df['High'].values, np.greater,
                                 order=self.window)[0]
        low_idx = argrelextrema(self.df['Low'].values, np.less,
                                order=self.window)[0]

        swing_highs = []
        swing_lows = []

        # Validate swing points with volume
        for idx in high_idx:
            if (self.df['Volume'].iloc[idx] >
                    self.df['Volume_MA'].iloc[idx] * self.volume_factor):
                swing_highs.append(self.df['High'].iloc[idx])

        for idx in low_idx:
            if (self.df['Volume'].iloc[idx] >
                    self.df['Volume_MA'].iloc[idx] * self.volume_factor):
                swing_lows.append(self.df['Low'].iloc[idx])

        return np.array(swing_highs), np.array(swing_lows)

    def _cluster_levels(self, levels):
        """Cluster nearby price levels using DBSCAN"""
        if len(levels) == 0:
            return []

        # Normalize prices for clustering
        prices = levels.reshape(-1, 1)
        mean_price = np.mean(levels)
        eps = mean_price * self.min_level_distance

        # Perform DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=1).fit(prices)

        # Calculate cluster centers
        unique_labels = np.unique(clustering.labels_)
        clustered_levels = []

        for label in unique_labels:
            mask = clustering.labels_ == label
            cluster_prices = levels[mask]

            # Calculate weighted average based on frequency of touches
            unique_prices, counts = np.unique(cluster_prices, return_counts=True)
            weighted_avg = np.average(unique_prices, weights=counts)
            clustered_levels.append(weighted_avg)

        return np.array(clustered_levels)

    def _calculate_level_strength(self, level, tolerance):
        """Calculate the strength of a price level based on touches and rejections"""
        touches = 0
        strong_touches = 0

        for idx in range(len(self.df)):
            high, low = self.df['High'].iloc[idx], self.df['Low'].iloc[idx]
            volume = self.df['Volume'].iloc[idx]
            volume_ma = self.df['Volume_MA'].iloc[idx]

            # Check if price touched the level
            if (abs(high - level) <= tolerance * level or
                    abs(low - level) <= tolerance * level):
                touches += 1

                # Strong touch if accompanied by high volume
                if volume > volume_ma * self.volume_factor:
                    strong_touches += 1

        return touches + (strong_touches * 0.5)  # Weight strong touches more

    def find_support_resistance(self, num_levels=5):
        """Find the most significant support and resistance levels"""
        # Get current price
        current_price = self.df['Close'].iloc[-1]

        # Find swing points
        swing_highs, swing_lows = self._find_swing_points()

        # Separate levels
        resistance_levels = swing_highs[swing_highs > current_price]
        support_levels = swing_lows[swing_lows < current_price]

        # Cluster levels
        resistance_clusters = self._cluster_levels(resistance_levels)
        support_clusters = self._cluster_levels(support_levels)

        # Calculate strength for each level
        resistance_strength = [(level, self._calculate_level_strength(level, self.touch_threshold))
                               for level in resistance_clusters]
        support_strength = [(level, self._calculate_level_strength(level, self.touch_threshold))
                            for level in support_clusters]

        # Sort by strength and distance from current price
        resistance_strength.sort(key=lambda x: (-x[1], x[0]))
        support_strength.sort(key=lambda x: (-x[1], -x[0]))

        # Get top levels
        top_resistance = [level for level, _ in resistance_strength[:num_levels]]
        top_support = [level for level, _ in support_strength[:num_levels]]

        return {
            'current_price': current_price,
            'support_levels': top_support,
            'resistance_levels': top_resistance
        }

    def get_level_statistics(self, level, window=100):
        """Get statistics for a specific price level"""
        recent_data = self.df.tail(window)
        tolerance = level * self.touch_threshold

        touches = sum(1 for idx in range(len(recent_data))
                      if abs(recent_data['High'].iloc[idx] - level) <= tolerance or
                      abs(recent_data['Low'].iloc[idx] - level) <= tolerance)

        rejection_strength = sum(
            recent_data['Volume'].iloc[idx] / recent_data['Volume_MA'].iloc[idx]
            for idx in range(len(recent_data))
            if abs(recent_data['High'].iloc[idx] - level) <= tolerance or
            abs(recent_data['Low'].iloc[idx] - level) <= tolerance
        )

        return {
            'touches': touches,
            'rejection_strength': rejection_strength,
            'avg_volume_ratio': rejection_strength / touches if touches > 0 else 0
        }


# Example usage
if __name__ == "__main__":
    file_path = "BNBUSDT.csv"  # Replace with your CSV file path

    # Initialize analyzer
    analyzer = PriceLevelAnalyzer(
        file_path,
        window=20,
        touch_threshold=0.002,
        volume_factor=1.5,
        min_level_distance=0.01
    )

    # Get support and resistance levels
    results = analyzer.find_support_resistance(num_levels=5)

    print(f"\nCurrent Price: {results['current_price']:.2f}")

    print("\nSupport Levels:")
    for i, level in enumerate(results['support_levels'], 1):
        stats = analyzer.get_level_statistics(level)
        print(f"S{i}: {level:.2f} (Touches: {stats['touches']}, "
              f"Strength: {stats['rejection_strength']:.2f})")

    print("\nResistance Levels:")
    for i, level in enumerate(results['resistance_levels'], 1):
        stats = analyzer.get_level_statistics(level)
        print(f"R{i}: {level:.2f} (Touches: {stats['touches']}, "
              f"Strength: {stats['rejection_strength']:.2f})")