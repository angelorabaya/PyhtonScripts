#import os
#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt


# Load and preprocess the data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')  # Sort in ascending order
    return df


# Prepare data for LSTM
def prepare_data(data, look_back=60, future_steps=30):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])

    X, y = [], []
    for i in range(len(scaled_data) - look_back - future_steps + 1):
        X.append(scaled_data[i:(i + look_back)])
        y.append(scaled_data[i + look_back:i + look_back + future_steps, 3])  # Predicting future Close prices

    return np.array(X), np.array(y), scaler


# Build the enhanced BreakGPT-inspired model
def build_model(input_shape, output_steps):
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True, input_shape=input_shape)),
        BatchNormalization(),
        Dropout(0.3),
        Bidirectional(LSTM(128, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.3),
        Bidirectional(LSTM(64, return_sequences=False)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(output_steps)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model


# Predict future prices
def predict_future(model, last_sequence, scaler, num_days=30):
    future_predictions = model.predict(last_sequence.reshape(1, *last_sequence.shape))[0]
    # Reshape predictions to match scaler's expected input
    future_predictions_reshaped = np.zeros((future_predictions.shape[0], 5))
    future_predictions_reshaped[:, 3] = future_predictions  # Set Close price column
    return scaler.inverse_transform(future_predictions_reshaped)[:, 3]


# Evaluate model
def evaluate_model(model, X_test, y_test, scaler):
    predictions = model.predict(X_test)
    # Reshape predictions and y_test to match scaler's expected input
    predictions_reshaped = np.zeros((predictions.shape[0] * predictions.shape[1], 5))
    predictions_reshaped[:, 3] = predictions.flatten()
    y_test_reshaped = np.zeros((y_test.shape[0] * y_test.shape[1], 5))
    y_test_reshaped[:, 3] = y_test.flatten()

    predictions_inv = scaler.inverse_transform(predictions_reshaped)[:, 3]
    y_test_inv = scaler.inverse_transform(y_test_reshaped)[:, 3]

    mse = np.mean((predictions_inv - y_test_inv) ** 2)
    rmse = np.sqrt(mse)
    return rmse


# Plot results
def plot_results(df, predictions_df):
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Close'], label='Historical Close Price')
    plt.plot(predictions_df['Date'], predictions_df['Predicted_Close'], label='Predicted Close Price')
    plt.title('Cryptocurrency Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


# Main function
def main(file_path):
    # Load and prepare data
    df = load_data(file_path)
    look_back = 60
    future_steps = 30
    X, y, scaler = prepare_data(df, look_back, future_steps)

    # Split data using TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    # Build and train model
    model = build_model((X.shape[1], X.shape[2]), future_steps)

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    # Evaluate model
    rmse = evaluate_model(model, X_test, y_test, scaler)
    print(f"Root Mean Squared Error: {rmse}")

    # Predict future prices
    last_sequence = X[-1]
    future_prices = predict_future(model, last_sequence, scaler, num_days=future_steps)

    # Generate future dates
    last_date = df['Date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_steps)

    # Create DataFrame with predictions
    predictions_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Close': future_prices
    })

    print("Future price predictions:")
    print(predictions_df)

    # Plot results
    #plot_results(df, predictions_df)


if __name__ == "__main__":
    file_path = "BTCUSDT.csv"  # Replace with your CSV file path
    main(file_path)