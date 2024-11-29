import requests
import pandas as pd
from datetime import datetime
import time


class CryptoDataAggregator:
    def __init__(self):
        self.coingecko_api = "https://api.coingecko.com/api/v3"
        self.etherscan_api = "CG-WNFRmP6DvNopNMqZmmwEGJAN"  # Get from etherscan.io
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    def get_market_data(self, coin_id):
        try:
            url = f"{self.coingecko_api}/simple/price"
            params = {
                'ids': coin_id,
                'vs_currencies': 'usd',
                'include_24hr_vol': True,
                'include_30d_change': True,
                'include_market_cap': True
            }
            response = requests.get(url, params=params, headers=self.headers)
            data = response.json()

            market_metrics = pd.DataFrame({
                'Current Price (USD)': [data[coin_id]['usd']],
                'Market Cap': [data[coin_id]['usd_market_cap']],
                'Volume (24h)': [data[coin_id]['usd_24h_vol']],
                'Price Change (30d)': [data[coin_id].get('usd_30d_change', 0)]
            })

            return market_metrics
        except Exception as e:
            print(f"Error fetching market data: {e}")
            return pd.DataFrame()

    def get_onchain_metrics(self, coin_id):
        try:
            url = f"{self.coingecko_api}/coins/{coin_id}"
            response = requests.get(url, headers=self.headers)
            data = response.json()

            supply_metrics = pd.DataFrame({
                'Circulating Supply': [data['market_data']['circulating_supply']],
                'Total Supply': [data['market_data']['total_supply']],
                'Max Supply': [data['market_data']['max_supply']],
                'Fully Diluted Valuation': [data['market_data'].get('fully_diluted_valuation', {}).get('usd', 0)]
            })

            return supply_metrics
        except Exception as e:
            print(f"Error fetching on-chain metrics: {e}")
            return pd.DataFrame()

    def get_github_metrics(self, coin_id):
        try:
            url = f"{self.coingecko_api}/coins/{coin_id}/developer_data"
            response = requests.get(url, headers=self.headers)
            data = response.json()

            dev_metrics = pd.DataFrame({
                'GitHub Stars': [data.get('stars', 0)],
                'GitHub Forks': [data.get('forks', 0)],
                'GitHub Issues': [data.get('total_issues', 0)],
                'GitHub PRs': [data.get('pull_requests_merged', 0)],
                'GitHub Contributors': [data.get('contributors', 0)]
            })

            return dev_metrics
        except Exception as e:
            print(f"Error fetching GitHub metrics: {e}")
            return pd.DataFrame()

    def calculate_nvt_ratio(self, market_cap, volume):
        try:
            nvt = market_cap / volume if volume > 0 else 0
            return pd.DataFrame({
                'NVT Ratio': [nvt],
                'Market Cap to Volume Ratio': [nvt]
            })
        except Exception as e:
            print(f"Error calculating NVT ratio: {e}")
            return pd.DataFrame()

    def get_fundamental_score(self, market_metrics, supply_metrics, nvt_ratio):
        try:
            # Simple scoring system (you can make this more sophisticated)
            score = 0
            if market_metrics['Price Change (30d)'].values[0] > 0:
                score += 1
            if nvt_ratio['NVT Ratio'].values[0] > 0 and nvt_ratio['NVT Ratio'].values[0] < 20:
                score += 1
            if supply_metrics['Circulating Supply'].values[0] > 0:
                score += 1

            rating = "Strong" if score >= 2 else "Moderate" if score >= 1 else "Weak"

            return pd.DataFrame({
                'Overall Score (0-3)': [score],
                'Rating': [rating],
                'Analysis Time': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
            })
        except Exception as e:
            print(f"Error calculating fundamental score: {e}")
            return pd.DataFrame()


def main():
    aggregator = CryptoDataAggregator()

    # List of coins to analyze
    coins = ['bitcoin', 'ethereum']

    for coin in coins:
        print(f"\nAnalyzing {coin.upper()}...")

        # Get all metrics
        market_metrics = aggregator.get_market_data(coin)
        time.sleep(1)  # Respect API rate limits

        supply_metrics = aggregator.get_onchain_metrics(coin)
        time.sleep(1)

        dev_metrics = aggregator.get_github_metrics(coin)
        time.sleep(1)

        # Calculate NVT ratio
        if not market_metrics.empty:
            nvt_ratio = aggregator.calculate_nvt_ratio(
                market_metrics['Market Cap'].values[0],
                market_metrics['Volume (24h)'].values[0]
            )
        else:
            nvt_ratio = pd.DataFrame()

        # Calculate fundamental score
        fundamental_analysis = aggregator.get_fundamental_score(
            market_metrics, supply_metrics, nvt_ratio
        )

        # Print results
        print("\nMarket Data:")
        print(market_metrics)
        print("\nOn-Chain Metrics:")
        print(supply_metrics)
        print("\nNetwork Value Metrics:")
        print(nvt_ratio)
        print("\nDevelopment Activity:")
        print(dev_metrics)
        print("\nFundamental Analysis:")
        print(fundamental_analysis)

        print("\n" + "=" * 50)


if __name__ == "__main__":
    main()