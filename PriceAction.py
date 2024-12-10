import pandas as pd

def read_data(file_path):
    # Read the CSV file
    data = pd.read_csv(file_path)
    # Ensure the Date column is of datetime type and sort in descending order
    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values(by='Date', ascending=False, inplace=True)
    return data

def predict_next_candle(data):
    # Calculate the simple moving average of the last few closing prices
    close_prices = data['Close'].values
    # Use the last 5 closing prices to predict the next one
    if len(close_prices) < 5:
        print("Not enough data to make a prediction.")
        return None

    # Simple prediction: average of the last 5 closes
    predicted_close = close_prices[:5].mean()
    return predicted_close

def main():
    file_path = 'SOLUSDT.csv'  # Replace with your actual CSV file path
    data = read_data(file_path)

    predicted_close = predict_next_candle(data)

    if predicted_close is not None:
        print(f'Predicted Next Candle Close Price: {predicted_close:.2f}')

if __name__ == "__main__":
    main()