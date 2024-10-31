import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Load the dataset
data = pd.read_csv('BTC_USDT.csv')

# Preprocess the data
data['Date'] = pd.to_datetime(data['Date'])  # Convert date to datetime
data.set_index('Date', inplace=True)  # Set Date as index
data = data[['High', 'Low', 'Open', 'Volumefrom', 'Volumeto', 'Close']]  # Select relevant features

# Feature engineering: Create additional features if necessary
data['Price_Change'] = data['Close'].shift(-1) - data['Close']  # Target variable

# Drop rows with NaN values
data.dropna(inplace=True)

# Define features and target
X = data.drop(columns=['Price_Change'])
y = data['Price_Change']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the XGBoost model
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

# Generate multiple future predictions
future_steps = 10  # Number of future steps to predict
predicted_prices = []

# Use the last row of the dataset as the starting point
last_row = X.iloc[-1].values.reshape(1, -1)  # Get the last row as a numpy array
last_close = data['Close'].iloc[-1]

for _ in range(future_steps):
    # Predict the next price change
    next_price_change = model.predict(last_row)

    # Calculate the next predicted close price
    last_close += next_price_change[0]
    predicted_prices.append(last_close)

    # Update last_row with the new Close price
    last_row[0, -1] = last_close

# Create a DataFrame for future prices
future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=future_steps, freq='D')
future_prices = pd.DataFrame({'Predicted_Close': predicted_prices}, index=future_dates)

# Print future prices
print("Future Predicted Prices:")
print(future_prices)