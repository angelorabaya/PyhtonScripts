import requests
import time
from datetime import datetime, timedelta
import pandas as pd


class CryptoLiquidationTracker:
    def __init__(self):
        self.binance_futures_url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
        self.binance_liquidation_url = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
        self.bybit_liquidation_url = "https://api.bybit.com/v2/public/liquidation-orders"

    def get_binance_liquidations(self, symbol="BTCUSDT"):
        try:
            params = {
                "symbol": symbol,
                "period": "1h"  # Available periods: "5m","15m","30m","1h","2h","4h","6h","12h","1d"
            }

            response = requests.get(self.binance_liquidation_url, params=params)
            if response.status_code == 200:
                data = response.json()
                return data
            return None
        except Exception as e:
            print(f"Error fetching Binance liquidations: {e}")
            return None

    def get_bybit_liquidations(self, symbol="BTCUSDT"):
        try:
            params = {
                "symbol": symbol,
                "limit": 50
            }

            response = requests.get(self.bybit_liquidation_url, params=params)
            if response.status_code == 200:
                data = response.json()
                return data.get('result', [])
            return None
        except Exception as e:
            print(f"Error fetching Bybit liquidations: {e}")
            return None

    def get_aggregated_liquidation_data(self, currency):
        liquidation_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'binance': {
                'long_ratio': 0,
                'short_ratio': 0
            },
            'bybit': {
                'liquidations': []
            },
            'total_liquidations': 0
        }

        # Get Binance data
        binance_data = self.get_binance_liquidations(symbol=currency)
        if binance_data:
            try:
                latest_data = binance_data[-1]
                liquidation_data['binance']['long_ratio'] = float(latest_data.get('longAccount', 0))
                liquidation_data['binance']['short_ratio'] = float(latest_data.get('shortAccount', 0))
            except (IndexError, KeyError) as e:
                print(f"Error processing Binance data: {e}")

        # Get Bybit data
        bybit_data = self.get_bybit_liquidations(symbol=currency)
        if bybit_data:
            liquidation_data['bybit']['liquidations'] = bybit_data
            liquidation_data['total_liquidations'] += len(bybit_data)

        return liquidation_data

    def monitor_liquidations(self, interval=300):  # 5 minutes default
        while True:
            print("\nFetching liquidation data...")
            data = self.get_aggregated_liquidation_data()

            print(f"\nTimestamp: {data['timestamp']}")
            print("\nBinance Futures Data:")
            print(
                f"Long/Short Ratio - Long: {data['binance']['long_ratio']:.2f} Short: {data['binance']['short_ratio']:.2f}")

            print("\nBybit Liquidations:")
            for liq in data['bybit']['liquidations'][:5]:  # Show only last 5 liquidations
                if isinstance(liq, dict):
                    print(f"Symbol: {liq.get('symbol', 'N/A')} | "
                          f"Side: {liq.get('side', 'N/A')} | "
                          f"Quantity: {liq.get('qty', 'N/A')} | "
                          f"Price: {liq.get('price', 'N/A')}")

            print(f"\nTotal Liquidations: {data['total_liquidations']}")

            print(f"\nWaiting {interval} seconds for next update...")
            time.sleep(interval)