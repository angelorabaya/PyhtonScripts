import requests
import pandas as pd

# Replace with your own API key
api_key = 'Z796Q911XLQXDL3P'
url = f'https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol=BTCUSDT&apikey={api_key}&datatype=csv'

# Send GET request to the API
response = requests.get(url)

# Ensure the request was successful
if response.status_code == 200:
    # Create a DataFrame from the CSV data
    from io import StringIO
    data = StringIO(response.text)
    df = pd.read_csv(data)

    # Save the DataFrame to a CSV file
    df.to_csv('BTCUSDTM.csv', index=False)
    print("Data saved")
else:
    print(f"Failed to retrieve data: {response.status_code}")