import requests

def get_ethereum_whale_activity(address='0x742d35Cc6634C0532925a3b844Bc454e4438f44e', api_key='2WHUJR2DEXIGZYC7N8BM6XPPSCFTF34HHH'):
    try:
        # Get balance and transaction count for the Ethereum address
        url = f"https://api.etherscan.io/api?module=account&action=balance&address={address}&tag=latest&apikey={api_key}"
        response = requests.get(url)
        balance_data = response.json()

        # Get transaction count and list of transactions
        tx_url = f"https://api.etherscan.io/api?module=account&action=txlist&address={address}&startblock=0&endblock=99999999&sort=asc&apikey={api_key}"
        tx_response = requests.get(tx_url)
        tx_data = tx_response.json()

        if balance_data['status'] == '1' and tx_data['status'] == '1':
            return {
                'final_balance': int(balance_data['result']) / 10**18,  # Convert Wei to ETH
                'n_tx': len(tx_data['result'])  # Total number of transactions
            }
        else:
            return {"error": "Failed to fetch data from Etherscan"}

    except Exception as e:
        print(f"Error fetching Ethereum whale activity: {str(e)}")
        return None