import pandas as pd
import numpy as np
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.layers import Input

# Load the data
data = pd.read_csv('SOLUSDT.csv', parse_dates=['Date'], index_col='Date')
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

# Future Prediction

# Number of future time steps to predict
future_steps = 30

# Get the last 'time_step' data points from the training set
last_sequence = X_train[-1]

# Store the predictions
future_predictions = []

# Predict future steps iteratively
for _ in range(future_steps):
    # Reshape the last sequence to 3D
    last_sequence = last_sequence.reshape(1, time_step, 1)

    # Make the prediction
    next_step_prediction = model.predict(last_sequence)

    # Append the prediction to the list
    future_predictions.append(next_step_prediction[0, 0])

    # Update the last sequence for the next prediction
    last_sequence = np.append(last_sequence[:, 1:, :], next_step_prediction[:, np.newaxis], axis=1)

# Inverse transform the predictions to get actual prices
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Create future dates for plotting
future_dates = pd.date_range(start=data.index[-1], periods=future_steps + 1, freq='h')[1:]

print(future_predictions)