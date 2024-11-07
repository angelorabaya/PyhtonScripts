import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import timedelta

# Load the data
data = pd.read_csv('DOGEUSDT.csv')

# Convert 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])
data.sort_values('Date', ascending=False, inplace=True)

# Create features and target
data['Target'] = data['Close'].shift(-1)  # Shift Close prices for prediction
data.dropna(inplace=True)  # Remove last row with NaN target

# Feature set
features = data[['Open', 'High', 'Low', 'Close', 'Volume']]
target = data['Target']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize and fit the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Predict future prices
last_row = features.iloc[-1].values.reshape(1, -1)  # Last known features
future_prices = []

for _ in range(10):  # Predict for the next 10 days
    future_price = model.predict(last_row)
    future_prices.append(future_price[0])
    last_row = np.array([[future_price[0], last_row[0][1], last_row[0][2], future_price[0], last_row[0][4]]])

# Output the future prices
future_dates = [data['Date'].max() + timedelta(days=i) for i in range(1, 11)]
future_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_prices})
print(future_df)