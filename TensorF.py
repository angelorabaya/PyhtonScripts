import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv('DJT.csv')
# Assuming 'Close' is the column with price data
prices = data['Close'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(prices)

# Function to prepare the data
def create_dataset(data, time_step=1, days_to_predict=7):
    X, y = [], []
    for i in range(len(data) - time_step - days_to_predict + 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[(i + time_step):(i + time_step + days_to_predict), 0])
    return np.array(X), np.array(y)

# Define time step
time_step = 60

# Create dataset
X, y = create_dataset(scaled_data, time_step, 7)

# Split into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50))
model.add(Dense(7))  # Output layer for 7 days prediction

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Make predictions
predictions = model.predict(X_test)

# Inverse transform predictions to get actual prices
predictions = scaler.inverse_transform(predictions)

# Print the 7 days future prices
print("7 Days Future Prices:")
for i, price in enumerate(predictions[0]):  # Just using predictions[0]
    print(f"Day {i + 1}: ${price:.2f}")