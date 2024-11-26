import requests
import pandas as pd
from tabulate import tabulate

url = "https://api.coingecko.com/api/v3/coins/list"
headers = {"accept": "application/json"}

response = requests.get(url, headers=headers)
data = response.json()

# Convert the data to a pandas DataFrame
df = pd.DataFrame(data)

# Display the data using tabulate
print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))