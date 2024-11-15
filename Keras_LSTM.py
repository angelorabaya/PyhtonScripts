import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from datetime import timedelta


def load_and_prepare_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Sort the dataframe by Date in ascending order
    df = df.sort_values('Date', ascending=True)

    # Set Date as index
    df = df.set_index('Date')
    return df


def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length, 0])  # Only predict Close price
    return np.array(X), np.array(y)


def build_lstm_model(sequence_length, n_features):
    model = Sequential()
    model.add(Input(shape=(sequence_length, n_features)))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mse')
    return model


def predict_future_prices(model, last_sequence, n_steps, price_scaler):
    future_predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(n_steps):
        # Get prediction for next price
        next_pred = model.predict(current_sequence.reshape(1, sequence_length, n_features))
        future_predictions.append(next_pred[0, 0])

        # Update sequence for next prediction
        current_sequence = np.roll(current_sequence, -1, axis=0)
        #current_sequence[-1, 0] = next_pred  # Update only the price
        current_sequence[-1, 0] = next_pred[0, 0]
        current_sequence[-1, 1] = current_sequence[-2, 1]  # Keep the last volume

    # Inverse transform predictions
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = price_scaler.inverse_transform(future_predictions)

    return future_predictions


# Main execution
if __name__ == "__main__":
    # Parameters
    file_path = 'BTCUSDT.csv'  # Replace with your CSV file path
    sequence_length = 60  # Number of time steps to look back
    future_days = 30  # Number of days to predict

    try:
        # Load and prepare data
        df = load_and_prepare_data(file_path)

        print(f"\nHistorical data range:")
        print(f"Start date: {df.index[0]}")
        print(f"End date: {df.index[-1]}")

        # Prepare features
        data = df[['Close', 'Volume']].copy()

        # Scale the features separately
        price_scaler = MinMaxScaler()
        volume_scaler = MinMaxScaler()

        scaled_close = price_scaler.fit_transform(data[['Close']])
        scaled_volume = volume_scaler.fit_transform(data[['Volume']])

        # Combine scaled features
        scaled_data = np.column_stack((scaled_close, scaled_volume))

        # Create sequences for training
        n_features = scaled_data.shape[1]
        X, y = create_sequences(scaled_data, sequence_length)

        # Split data into training and testing sets
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Build and train the model
        model = build_lstm_model(sequence_length, n_features)
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            verbose=1
        )

        # Prepare last sequence for future prediction
        last_sequence = scaled_data[-sequence_length:]

        # Predict future prices
        future_predictions = predict_future_prices(model, last_sequence, future_days, price_scaler)

        # Generate future dates (excluding weekends)
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1),
                                     periods=future_days,
                                     freq='B')

        # Create prediction dataframe
        prediction_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Price': np.round(future_predictions.flatten(), 2)
        })

        print("\nPredicted Prices (Next 30 Business Days):")
        print(prediction_df.to_string())

    except Exception as e:
        print(f"An error occurred: {str(e)}")