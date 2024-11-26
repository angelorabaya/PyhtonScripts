import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt


def analyze_price_levels(file_path):
    # Read CSV file
    df = pd.read_csv(file_path)

    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Sort by date ascending for proper analysis
    df = df.sort_values('Date')

    # Calculate additional technical indicators
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['ATR'] = calculate_atr(df)

    # Find support and resistance levels using local minima and maxima
    price_series = df['Close'].values
    n = 20  # Increased window size for more significant levels

    # Find local maxima and minima
    local_max_indices = argrelextrema(price_series, np.greater, order=n)[0]
    local_min_indices = argrelextrema(price_series, np.less, order=n)[0]

    resistance_levels = price_series[local_max_indices]
    support_levels = price_series[local_min_indices]

    # Filter significant levels using improved clustering
    significant_resistance = get_significant_levels(resistance_levels, max_levels=3)
    significant_support = get_significant_levels(support_levels, max_levels=3)

    # Calculate potential take profit and stop loss levels
    current_price = df['Close'].iloc[-1]
    atr = df['ATR'].iloc[-1]

    take_profit_levels = calculate_take_profit(current_price, significant_resistance)
    stop_loss_levels = calculate_stop_loss(current_price, significant_support, atr)

    return {
        'support_levels': significant_support,
        'resistance_levels': significant_resistance,
        'take_profit_levels': take_profit_levels,
        'stop_loss_levels': stop_loss_levels,
        'current_price': current_price,
        'df': df
    }


def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)

    return true_range.rolling(period).mean()


def get_significant_levels(levels, max_levels=3, tolerance=0.02):
    if len(levels) == 0:
        return np.array([])

    # Sort levels
    levels = np.sort(levels)

    # Group close levels together
    clusters = []
    current_cluster = [levels[0]]

    for i in range(1, len(levels)):
        if (levels[i] - current_cluster[-1]) / current_cluster[-1] <= tolerance:
            current_cluster.append(levels[i])
        else:
            clusters.append(np.mean(current_cluster))
            current_cluster = [levels[i]]

    clusters.append(np.mean(current_cluster))
    clusters = np.array(clusters)

    # Select most significant levels based on price distribution
    if len(clusters) <= max_levels:
        return np.array(clusters)

    # For resistance, prioritize recent and higher levels
    # Space them out evenly across the price range
    indices = np.linspace(0, len(clusters) - 1, max_levels, dtype=int)
    return clusters[indices]


def calculate_take_profit(current_price, resistance_levels):
    # Filter resistance levels above current price
    potential_targets = resistance_levels[resistance_levels > current_price]

    if len(potential_targets) == 0:
        # If no resistance above, use percentage-based targets
        return np.array([current_price * (1 + x / 100) for x in [2, 5, 10]])

    return potential_targets[:min(3, len(potential_targets))]


def calculate_stop_loss(current_price, support_levels, atr):
    # Dynamic stop loss based on ATR and nearest support
    atr_stop = current_price - (2 * atr)

    # Filter support levels below current price
    potential_stops = support_levels[support_levels < current_price]

    if len(potential_stops) == 0:
        return np.array([atr_stop])

    # Take the highest support levels below current price
    stops = potential_stops[-min(2, len(potential_stops)):]
    # Add ATR-based stop
    stops = np.append(stops, atr_stop)
    return np.sort(stops)[::-1]  # Sort in descending order


def plot_analysis(analysis_results):
    df = analysis_results['df']

    plt.figure(figsize=(15, 7))

    # Plot price and moving averages
    plt.plot(df['Date'], df['Close'], label='Price', color='blue')
    plt.plot(df['Date'], df['SMA20'], label='SMA 20', color='orange', alpha=0.7)
    plt.plot(df['Date'], df['SMA50'], label='SMA 50', color='red', alpha=0.7)

    # Plot support levels
    for i, level in enumerate(analysis_results['support_levels'], 1):
        plt.axhline(y=level, color='green', linestyle='--', alpha=0.5,
                    label=f'Support {i}: {level:.2f}')

    # Plot resistance levels
    for i, level in enumerate(analysis_results['resistance_levels'], 1):
        plt.axhline(y=level, color='red', linestyle='--', alpha=0.5,
                    label=f'Resistance {i}: {level:.2f}')

    # Plot current price
    plt.axhline(y=analysis_results['current_price'], color='black', linestyle='-.',
                alpha=0.5, label=f'Current: {analysis_results["current_price"]:.2f}')

    plt.title('Price Analysis with Key Support/Resistance Levels')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def print_analysis_results(analysis_results):
    print("\nAnalysis Results:")
    print(f"Current Price: {analysis_results['current_price']:.2f}")

    print("\nKey Support Levels:")
    for i, level in enumerate(analysis_results['support_levels'], 1):
        print(f"Support {i}: {level:.2f}")

    print("\nKey Resistance Levels:")
    for i, level in enumerate(analysis_results['resistance_levels'], 1):
        print(f"Resistance {i}: {level:.2f}")

    print("\nRecommended Take Profit Levels:")
    for i, level in enumerate(analysis_results['take_profit_levels'], 1):
        print(f"TP {i}: {level:.2f}")

    print("\nRecommended Stop Loss Levels:")
    for i, level in enumerate(analysis_results['stop_loss_levels'], 1):
        print(f"SL {i}: {level:.2f}")


# Example usage
if __name__ == "__main__":
    try:
        # Replace with your CSV file path
        file_path = "WOOUSDT.csv"

        # Perform analysis
        analysis_results = analyze_price_levels(file_path)

        # Print results
        print_analysis_results(analysis_results)

        # Plot the analysis
        plot_analysis(analysis_results)

    except FileNotFoundError:
        print("Error: CSV file not found. Please check the file path.")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Please ensure your CSV file has the correct columns: Date,Open,High,Low,Close,Volume")