import pandas as pd
import requests
from datetime import datetime, timedelta

class CryptoFundamentalAnalysis:
    def __init__(self, crypto_id):
        self.crypto_id = crypto_id
        self.coingecko_api = "https://api.coingecko.com/api/v3"
        self.etherscan_api = "CG-WNFRmP6DvNopNMqZmmwEGJAN"  # Replace with your API key or CG-WNFRmP6DvNopNMqZmmwEGJAN / 2WHUJR2DEXIGZYC7N8BM6XPPSCFTF34HHH

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

            #return pd.DataFrame.from_dict(market_data, orient='index')
            return market_data

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

            #return pd.DataFrame.from_dict(metrics, orient='index')
            return metrics

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

            #return pd.DataFrame.from_dict(metrics, orient='index')
            return metrics

        except Exception as e:
            return f"Error calculating network value metrics: {str(e)}"

    def get_development_activity(self):
        try:
            # GitHub API endpoints
            headers = {
                'Authorization': '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',  # Create personal access token on GitHub
                'Accept': 'application/vnd.github.v3+json'
            }

            # Repository mappings
            repo_mappings = {
                'bitcoin': 'bitcoin/bitcoin',
                'ethereum': 'ethereum/go-ethereum',
                'binancecoin': 'binance-chain/bsc',
                'cardano': 'input-output-hk/cardano-node',
                'solana': 'solana-labs/solana',
                'polkadot': 'paritytech/polkadot',
                'dogecoin': 'dogecoin/dogecoin',
                'litecoin': 'litecoin-project/litecoin',
                'chainlink': 'smartcontractkit/chainlink',
                'uniswap': 'Uniswap/v3-core',
                'aave': 'aave/aave-protocol',
                'ripple': 'ripple/rippled',
                'maker': 'makerdao/dss',
                'filecoin': 'filecoin-project/lotus',
                'tezos': 'tezos/tezos',
                'avalanche': 'ava-labs/avalanchego',
                'cosmos': 'cosmos/gaia',
                'monero': 'monero-project/monero',
                'zcash': 'zcash/zcash',
                'dash': 'dashpay/dash',
                'algorand': 'algorand/go-algorand',
                'vechain': 'vechain/thor',
                'hedera': 'hashgraph/hedera-services',
                'internet-computer': 'dfinity/ic',
                'arbitrum': 'OffchainLabs/arbitrum',
                'optimism': 'ethereum-optimism/optimism',
                'decentraland': 'decentraland/contracts',
                'the-sandbox': 'thesandboxgame/sandbox-smart-contracts',
                'axie-infinity': 'AxieInfinity/ronin',
                'immutable-x': 'ImmutableX/imx-core-sdk',
                'loopring': 'Loopring/loopring',
                'sushiswap': 'sushiswap/sushiswap',
                'curve-dao-token': 'curvefi/curve-dao-contracts'
            }

            if self.crypto_id not in repo_mappings:
                return pd.DataFrame({
                    'GitHub Stars': [0],
                    'GitHub Forks': [0],
                    'GitHub Issues': [0],
                    'GitHub Subscribers': [0],
                    'GitHub Contributors': [0]
                })

            repo = repo_mappings[self.crypto_id]
            base_url = f"https://api.github.com/repos/{repo}"

            # Get repository stats
            repo_response = requests.get(base_url, headers=headers)
            repo_data = repo_response.json()

            # Get contributors count
            contributors_response = requests.get(f"{base_url}/contributors?per_page=1&anon=true", headers=headers)
            contributors_count = len(
                requests.get(f"{base_url}/contributors?per_page=100&anon=true", headers=headers).json())

            # Get open issues count
            issues_response = requests.get(f"{base_url}/issues?state=open", headers=headers)
            issues_data = issues_response.json()

            dev_metrics = {
                'Development Metrics': {
                    'GitHub Stars': repo_data.get('stargazers_count', 0),
                    'GitHub Forks': repo_data.get('forks_count', 0),
                    'GitHub Issues': repo_data.get('open_issues_count', 0),
                    'GitHub Subscribers': repo_data.get('subscribers_count', 0),
                    'GitHub Contributors': contributors_count
                }
            }

            #return pd.DataFrame.from_dict(dev_metrics, orient='index')
            return dev_metrics

        except Exception as e:
            print(f"Error fetching development metrics: {str(e)}")
            return pd.DataFrame({
                'GitHub Stars': [0],
                'GitHub Forks': [0],
                'GitHub Issues': [0],
                'GitHub Subscribers': [0],
                'GitHub Contributors': [0]
            })

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

            #return pd.DataFrame.from_dict(analysis, orient='index')
            return analysis

        except Exception as e:
            return f"Error in fundamental analysis: {str(e)}"


def main():
    # Example usage
    crypto_id = "litecoin"
    cfa = CryptoFundamentalAnalysis(crypto_id)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def format_section(title, content):
        separator = "=" * 50
        return f"\n{separator}\n{title}\n{separator}\n{content}\n"

    print("CRYPTOCURRENCY ANALYSIS")
    print(f"Asset: {crypto_id.upper()}")
    print(cfa.get_market_data())
    print(cfa.get_on_chain_metrics())
    print(cfa.calculate_network_value_metrics())
    print(cfa.get_development_activity())

if __name__ == "__main__":
    main()