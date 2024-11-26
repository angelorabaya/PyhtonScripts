import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings

warnings.filterwarnings('ignore')


def load_and_prepare_data(csv_file):
    try:
        df = pd.read_csv(csv_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        # Technical indicators
        df['MA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
        df['MA50'] = df['Close'].rolling(window=50, min_periods=1).mean()
        df['RSI'] = calculate_rsi(df['Close'])
        df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'])
        df['MACD'] = calculate_macd(df['Close'])
        df['BB_Upper'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])
        df['Price_Change'] = df['Close'].pct_change()
        df['Volume_Change'] = df['Volume'].pct_change()

        # Advanced features
        df['Price_Range'] = df['High'] - df['Low']
        df['Price_Range_MA'] = df['Price_Range'].rolling(window=14, min_periods=1).mean()
        df['Volume_MA'] = df['Volume'].rolling(window=20, min_periods=1).mean()

        df = df.fillna(method='bfill')
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def calculate_rsi(prices, periods=14):
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods, min_periods=1).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs)).fillna(50)
    except Exception as e:
        return pd.Series([50] * len(prices))


def calculate_atr(high, low, close, period=14):
    try:
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.DataFrame({'TR1': tr1, 'TR2': tr2, 'TR3': tr3}).max(axis=1)
        return tr.rolling(window=period, min_periods=1).mean()
    except Exception as e:
        return pd.Series([0] * len(high))


def calculate_macd(prices, fast=12, slow=26, signal=9):
    try:
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        return macd
    except Exception as e:
        return pd.Series([0] * len(prices))


def calculate_bollinger_bands(prices, window=20, num_std=2):
    try:
        rolling_mean = prices.rolling(window=window, min_periods=1).mean()
        rolling_std = prices.rolling(window=window, min_periods=1).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, lower_band
    except Exception as e:
        return pd.Series([0] * len(prices)), pd.Series([0] * len(prices))


def prepare_lstm_data(df, lookback=60):
    try:
        features = ['Close', 'Volume', 'MA20', 'MA50', 'RSI', 'ATR', 'MACD',
                    'BB_Upper', 'BB_Lower', 'Price_Change', 'Volume_Change',
                    'Price_Range', 'Price_Range_MA', 'Volume_MA']

        price_scaler = MinMaxScaler()
        feature_scaler = MinMaxScaler()

        # Scale the target (Close price) separately
        prices_scaled = price_scaler.fit_transform(df[['Close']])
        features_scaled = feature_scaler.fit_transform(df[features])

        X, y = [], []
        for i in range(lookback, len(features_scaled)):
            X.append(features_scaled[i - lookback:i])
            y.append(prices_scaled[i])

        return np.array(X), np.array(y), price_scaler, feature_scaler
    except Exception as e:
        print(f"Error preparing LSTM data: {e}")
        return None, None, None, None


def build_advanced_lstm_model(input_shape):
    try:
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='huber', metrics=['mae'])
        return model
    except Exception as e:
        print(f"Error building LSTM model: {e}")
        return None


def predict_next_swing(df, model, price_scaler, feature_scaler, lookback=60):
    try:
        if len(df) < lookback:
            raise ValueError("Not enough data points for prediction")

        features = ['Close', 'Volume', 'MA20', 'MA50', 'RSI', 'ATR', 'MACD',
                    'BB_Upper', 'BB_Lower', 'Price_Change', 'Volume_Change',
                    'Price_Range', 'Price_Range_MA', 'Volume_MA']

        last_sequence = df[features].tail(lookback).values
        last_sequence_scaled = feature_scaler.transform(last_sequence)
        prediction_scaled = model.predict(np.array([last_sequence_scaled]))
        prediction = price_scaler.inverse_transform(prediction_scaled)[0][0]

        current_price = df['Close'].iloc[-1]
        price_change = prediction - current_price
        percentage_change = (price_change / current_price) * 100

        # Calculate confidence based on recent prediction accuracy
        recent_accuracy = calculate_prediction_accuracy(df, model, price_scaler, feature_scaler, lookback)

        return prediction, percentage_change, recent_accuracy
    except Exception as e:
        print(f"Error predicting next swing: {e}")
        return None, None, None


def calculate_prediction_accuracy(df, model, price_scaler, feature_scaler, lookback, window=10):
    try:
        features = ['Close', 'Volume', 'MA20', 'MA50', 'RSI', 'ATR', 'MACD',
                    'BB_Upper', 'BB_Lower', 'Price_Change', 'Volume_Change',
                    'Price_Range', 'Price_Range_MA', 'Volume_MA']

        recent_errors = []
        for i in range(window):
            if len(df) < lookback + i + 1:
                continue

            sequence = df[features].iloc[-(lookback + i + 1):-(i + 1)].values
            sequence_scaled = feature_scaler.transform(sequence)
            pred_scaled = model.predict(np.array([sequence_scaled]), verbose=0)
            pred_price = price_scaler.inverse_transform(pred_scaled)[0][0]
            actual_price = df['Close'].iloc[-(i + 1)]
            error = abs((pred_price - actual_price) / actual_price)
            recent_errors.append(1 - error)

        return np.mean(recent_errors) if recent_errors else 0
    except Exception as e:
        return 0


def main():
    try:
        csv_file = "ADAUSDT.csv"  # Replace with your CSV file path
        df = load_and_prepare_data(csv_file)

        if df is None or len(df) < 60:
            print("Insufficient data for analysis")
            return

        X, y, price_scaler, feature_scaler = prepare_lstm_data(df)

        if X is None or len(X) < 2:
            print("Insufficient data for LSTM analysis")
            return

        # Split data
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Build and train model
        model = build_advanced_lstm_model(input_shape=(X.shape[1], X.shape[2]))
        if model is None:
            return

        model.fit(X_train, y_train, epochs=50, batch_size=32,
                  validation_split=0.1, verbose=0)

        # Make prediction
        next_price, price_change, confidence = predict_next_swing(
            df, model, price_scaler, feature_scaler)

        if next_price is not None:
            current_price = df['Close'].iloc[-1]
            print("\nCurrent Price:", f"${current_price:.2f}")
            print("Predicted Next Price:", f"${next_price:.2f}")
            print("Expected Change:", f"{price_change:.2f}%")
            print("Prediction Confidence:", f"{confidence:.2%}")

            # Determine swing direction and strength
            swing_strength = abs(price_change)
            if price_change > 0:
                strength_label = "Strong Buy" if swing_strength > 2 else "Buy" if swing_strength > 1 else "Weak Buy"
            else:
                strength_label = "Strong Sell" if swing_strength > 2 else "Sell" if swing_strength > 1 else "Weak Sell"

            print("Signal:", strength_label)

            # Print support/resistance levels near predicted price
            price_range = df['Close'].max() - df['Close'].min()
            threshold = price_range * 0.02  # 2% of price range

            nearby_support = df[df['Close'] < next_price]['Close'].max()
            nearby_resistance = df[df['Close'] > next_price]['Close'].min()

            if abs(nearby_support - next_price) < threshold:
                print("Near Support Level:", f"${nearby_support:.2f}")
            if abs(nearby_resistance - next_price) < threshold:
                print("Near Resistance Level:", f"${nearby_resistance:.2f}")

    except Exception as e:
        print(f"Error in main execution: {e}")


if __name__ == "__main__":
    main()