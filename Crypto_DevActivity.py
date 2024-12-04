import pandas as pd
import requests

def get_development_activity(crypto_id):
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

        if crypto_id not in repo_mappings:
            return pd.DataFrame({
                'GitHub Stars': [0],
                'GitHub Forks': [0],
                'GitHub Issues': [0],
                'GitHub Subscribers': [0],
                'GitHub Contributors': [0]
            })

        repo = repo_mappings[crypto_id]
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

        # return pd.DataFrame.from_dict(dev_metrics, orient='index')
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