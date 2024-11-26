import pandas as pd
import requests
from datetime import datetime, timedelta
import numpy as np
from web3 import Web3
import time


class CryptoFundamentalAnalysis:
    def __init__(self, crypto_id):
        self.crypto_id = crypto_id
        self.coingecko_api = "https://api.coingecko.com/api/v3"
        self.etherscan_api = "CG-WNFRmP6DvNopNMqZmmwEGJAN"  # Replace with your API key

    def get_market_data(self):
        try:
            endpoint = f"{self.coingecko_api}/coins/{self.crypto_id}"
            response = requests.get(endpoint)
            data = response.json()

            market_data = {
                'Market Metrics': {
                    'Current Price (USD)': data['market_data']['current_price']['usd'],
                    'Market Cap (USD)': data['market_data']['market_cap']['usd'],
                    'Trading Volume (24h)': data['market_data']['total_volume']['usd'],
                    'Market Cap Rank': data['market_cap_rank'],
                    'Price Change (24h)': f"{data['market_data']['price_change_percentage_24h']}%",
                    'Price Change (7d)': f"{data['market_data']['price_change_percentage_7d']}%",
                    'Price Change (30d)': f"{data['market_data']['price_change_percentage_30d']}%"
                }
            }

            return pd.DataFrame.from_dict(market_data, orient='index')

        except Exception as e:
            return f"Error fetching market data: {str(e)}"

    def get_on_chain_metrics(self):
        try:
            endpoint = f"{self.coingecko_api}/coins/{self.crypto_id}"
            response = requests.get(endpoint)
            data = response.json()

            # Calculate additional metrics
            circulating_supply = data['market_data']['circulating_supply']
            total_supply = data['market_data']['total_supply']
            if total_supply is None:
                total_supply = circulating_supply

            supply_ratio = (circulating_supply / total_supply) if total_supply > 0 else 0

            metrics = {
                'Supply Metrics': {
                    'Circulating Supply': circulating_supply,
                    'Total Supply': total_supply,
                    'Supply Ratio': f"{supply_ratio:.2%}",
                    'Fully Diluted Valuation': data['market_data'].get('fully_diluted_valuation', {}).get('usd', 'N/A')
                }
            }

            return pd.DataFrame.from_dict(metrics, orient='index')

        except Exception as e:
            return f"Error fetching on-chain metrics: {str(e)}"

    def calculate_network_value_metrics(self):
        try:
            endpoint = f"{self.coingecko_api}/coins/{self.crypto_id}"
            response = requests.get(endpoint)
            data = response.json()

            market_cap = data['market_data']['market_cap']['usd']
            volume_24h = data['market_data']['total_volume']['usd']

            # Calculate NVT ratio (Network Value to Transactions)
            nvt_ratio = market_cap / volume_24h if volume_24h > 0 else 0

            metrics = {
                'Network Value Metrics': {
                    'NVT Ratio': f"{nvt_ratio:.2f}",
                    'Market Cap to Volume Ratio': f"{(market_cap / volume_24h):.2f}" if volume_24h > 0 else 'N/A'
                }
            }

            return pd.DataFrame.from_dict(metrics, orient='index')

        except Exception as e:
            return f"Error calculating network value metrics: {str(e)}"

    def get_development_activity(self):
        try:
            endpoint = f"{self.coingecko_api}/coins/{self.crypto_id}/developer_data"
            response = requests.get(endpoint)
            data = response.json()

            dev_metrics = {
                'Development Metrics': {
                    'GitHub Stars': data.get('stars', 0),
                    'GitHub Forks': data.get('forks', 0),
                    'GitHub Issues': data.get('total_issues', 0),
                    'GitHub Subscribers': data.get('subscribers', 0),
                    'GitHub Contributors': data.get('contributors', 0)
                }
            }

            return pd.DataFrame.from_dict(dev_metrics, orient='index')

        except Exception as e:
            return f"Error fetching development metrics: {str(e)}"

    def analyze_fundamentals(self):
        try:
            # Combine all analyses
            market_data = self.get_market_data()
            on_chain_metrics = self.get_on_chain_metrics()
            network_metrics = self.calculate_network_value_metrics()
            dev_metrics = self.get_development_activity()

            # Calculate fundamental score (basic implementation)
            score = 0

            # Market performance scoring
            if float(market_data.loc['Market Metrics', 'Price Change (24h)'].strip('%')) > 0:
                score += 1
            if float(market_data.loc['Market Metrics', 'Price Change (7d)'].strip('%')) > 0:
                score += 1

            # Volume analysis
            volume = float(market_data.loc['Market Metrics', 'Trading Volume (24h)'])
            market_cap = float(market_data.loc['Market Metrics', 'Market Cap (USD)'])
            if volume / market_cap > 0.1:  # Volume > 10% of market cap
                score += 1

            analysis = {
                'Fundamental Analysis': {
                    'Overall Score (0-3)': score,
                    'Rating': 'Strong' if score >= 2 else 'Moderate' if score == 1 else 'Weak',
                    'Analysis Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            }

            return pd.DataFrame.from_dict(analysis, orient='index')

        except Exception as e:
            return f"Error in fundamental analysis: {str(e)}"


def main():
    # Example usage
    crypto_id = "mav" #input("Enter cryptocurrency ID (e.g., bitcoin, ethereum): ")
    cfa = CryptoFundamentalAnalysis(crypto_id)

    print("\nMarket Data:")
    print(cfa.get_market_data())

    print("\nOn-Chain Metrics:")
    print(cfa.get_on_chain_metrics())

    print("\nNetwork Value Metrics:")
    print(cfa.calculate_network_value_metrics())

    print("\nDevelopment Activity:")
    print(cfa.get_development_activity())

    print("\nFundamental Analysis:")
    print(cfa.analyze_fundamentals())


if __name__ == "__main__":
    main()