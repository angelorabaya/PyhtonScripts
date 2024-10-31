import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

# Load data from CSV
data = pd.read_csv('BTCUSDT.csv')

# Feature selection
data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']] #Date,Open,High,Low,Close,Volume
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Prepare data
X = data[['High', 'Low', 'Open', 'Volume']].values
y = data['Close'].shift(-1).dropna().values  # Predicting next day's close price
X = X[:-1]  # Align X with y

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Scale the features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the Neural Network model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# Make predictions
predictions = model.predict(X_test)

# Display predictions
predicted_prices = pd.DataFrame(predictions, columns=['Predicted Price'])
predicted_prices['Actual Price'] = y_test

# Count bullish and bearish predictions
bullish_count = (predicted_prices['Predicted Price'] > predicted_prices['Actual Price']).sum()
bearish_count = (predicted_prices['Predicted Price'] < predicted_prices['Actual Price']).sum()

# Display counts
print(f'Bullish Predictions Count: {bullish_count}')
print(f'Bearish Predictions Count: {bearish_count}')

# Optionally: Evaluate the model
mse = model.evaluate(X_test, y_test)
print(f'Mean Squared Error: {mse}')