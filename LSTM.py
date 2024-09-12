import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
#from tensorflow.python.keras.saving.saved_model_experimental import sequential
from keras.layers import LSTM, Dense, Dropout
from keras.layers import Input

# Load the data
data = pd.read_csv('d:/BTCUSD.csv', parse_dates=['Date'], index_col='Date')
data = data[['Close']]  # Use closing prices for forecasting

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Prepare the data for LSTM
def create_dataset(dataset, time_step=1):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

# Create sequences
time_step = 60  # 60 minutes for time steps
X, y = create_dataset(scaled_data, time_step)

# Reshape X to be 3D [samples, time steps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split the dataset into training and testing
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build the LSTM model
model = Sequential()
model.add(Input(shape=(X_train.shape[1], 1)))  # Add Input layer
model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Plotting the results
plt.figure(figsize=(14,5))
plt.plot(data.index[train_size+time_step+1:], predictions, color='red', label='Predicted Price')
plt.plot(data.index, data['Close'], color='blue', label='Actual Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('BTC/USD Price Prediction')
plt.legend()
plt.show()