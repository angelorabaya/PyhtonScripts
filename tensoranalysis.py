import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from sklearn.preprocessing import MinMaxScaler
#import keras
#from keras import layers
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input

# Load the data
data = pd.read_csv('DJT.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Select 'Close' price
close_prices = data['Close'].values
close_prices = close_prices.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# Create training and testing datasets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Function to create the dataset for LSTM
def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        x = dataset[i:(i + time_step), 0]
        X.append(x)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

# Prepare the dataset
time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the LSTM model using Input layer
model = Sequential()
model.add(Input(shape=(X_train.shape[1], 1)))  # Explicitly define input shape
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=1, epochs=10)

# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Inverse transform the predictions

train_predictions = scaler.inverse_transform(train_predictions)
test_predictions = scaler.inverse_transform(test_predictions)

# Adjust the plotting to account for the lost data points due to time_step
plt.figure(figsize=(12, 6))

# Training data plot - adjust the end index
plt.plot(data.index[:train_size], close_prices[:train_size], label='Training Data')

# Predicted prices plot - adjust the start index and use the full range of test_predictions
plt.plot(data.index[train_size + time_step : train_size + time_step + len(test_predictions)],
         test_predictions, label='Predicted Prices', color='orange')

# Actual prices plot - adjust the start index
plt.plot(data.index[train_size + time_step:],
         close_prices[train_size + time_step:], label='Actual Prices', color='green')

plt.title('Bitcoin Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()