import pandas as pd
from prophet import Prophet

# Load the CSV file
def load_data(csv_file):
    data = pd.read_csv(csv_file)
    return data

# Prepare data for Prophet
def prepare_data(data):
    # Rename columns for Prophet: 'ds' for dates and 'y' for the values to predict
    data = data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    return data

# Train the Prophet model and make predictions
def predict_price(data, periods=30):
    model = Prophet()
    model.fit(data)

    # Create a dataframe for future dates
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def get_Prophet(csv_file):
    data = load_data(csv_file)
    prepared_data = prepare_data(data)
    predictions = predict_price(prepared_data)

    price_predict = predictions.tail(30).iloc[-1].iloc[1]
    close_prices = data['Close']
    last_close_price = close_prices.iloc[0]

    if price_predict > last_close_price:
        trend = "Bullish"
    else:
        trend = "Bearish"

    # Print the forecasted prices
    #print(predictions.tail(30))  # Print the last 30 rows of predictions

    return [predictions.tail(30), trend, last_close_price]

#if __name__ == "__main__":
#    main("BTCUSDT.csv")