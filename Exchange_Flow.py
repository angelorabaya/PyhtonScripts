import requests
import time
from datetime import datetime

class BlockchainMetricsTracker:
    def __init__(self):
        self.blockchair_base_url = "https://api.blockchair.com/bitcoin"
        self.etherscan_base_url = "https://api.etherscan.io/api"

    def get_metrics_data(self):
        try:
            metrics_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'blockchain_metrics': {},
                'top_addresses': []
            }

            # Get blockchain metrics
            metrics = self.get_blockchain_metrics()
            if metrics:
                metrics_data['blockchain_metrics'] = {
                    'transactions_24h': metrics.get('transactions_24h', 0),
                    'volume_24h': metrics.get('volume_24h', 0),
                    'blocks_24h': metrics.get('blocks_24h', 0),
                    'circulation': metrics.get('circulation', 0),
                    'average_transaction_fee_24h': metrics.get('average_transaction_fee_24h', 0),
                    'median_transaction_fee_24h': metrics.get('median_transaction_fee_24h', 0),
                    'mempool_transactions': metrics.get('mempool_transactions', 0),
                    'mempool_size': metrics.get('mempool_size', 0)
                }

            # Get top addresses
            addresses = self.get_top_addresses(limit=10)
            if addresses:
                metrics_data['top_addresses'] = [
                    {
                        'address': addr.get('address', ''),
                        'balance': addr.get('balance', 0),
                        'transaction_count': addr.get('transaction_count', 0),
                        'first_seen': addr.get('first_seen', ''),
                        'last_seen': addr.get('last_seen', '')
                    }
                    for addr in addresses
                ]

            return metrics_data

        except Exception as e:
            print(f"Error getting metrics data: {e}")
            return None

    def get_blockchain_metrics(self):
        """
        Get blockchain metrics from Blockchair
        """
        try:
            response = requests.get(f"{self.blockchair_base_url}/stats")
            if response.status_code == 200:
                return response.json().get('data', {})
            return None
        except Exception as e:
            print(f"Error fetching blockchain metrics: {e}")
            return None

    def get_top_addresses(self, limit=10):
        """
        Get top addresses by balance
        """
        try:
            response = requests.get(f"{self.blockchair_base_url}/addresses?limit={limit}")
            if response.status_code == 200:
                return response.json().get('data', [])
            return None
        except Exception as e:
            print(f"Error fetching top addresses: {e}")
            return None
