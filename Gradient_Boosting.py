import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import timedelta

# Load and preprocess the data
def load_data(file_path):
    df = pd.read_csv(file_path)
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    # Set Date as index
    df.set_index('Date', inplace=True)
    # Sort by date in ascending order
    df.sort_index(inplace=True)
    return df


# Prepare feature set and target variable
def prepare_data(df):
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    X = df[features]
    y = df['Close']
    return X, y


# Create and train GBM model
def create_gb_model(X_train, y_train):
    model = lgb.LGBMRegressor(
        objective='regression',
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=1000,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


# Make predictions
def make_predictions(model, X_test):
    predictions = model.predict(X_test)
    return predictions


# Main function
def main():
    # Load data
    df = load_data('BTCUSDT.csv')

    # Prepare feature set and target variable
    X, y = prepare_data(df)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and train GBM model
    model = create_gb_model(X_train_scaled, y_train)

    # Make predictions for next 7 days
    last_date = df.index[-1] + timedelta(days=6)
    future_dates = pd.date_range(start=df.index[-1], periods=7, freq='B')
    future_X = pd.DataFrame(index=future_dates, columns=X.columns)
    future_X.loc[:, 'Date'] = future_dates

    # Predict prices for future dates
    future_prices = model.predict(scaler.transform(future_X.drop(['Date'], axis=1)))

    # Print predicted prices
    print("Predicted prices for the next 7 days:")
    for i in range(7):
        print(f"{future_dates[i]}: ${future_prices[i]:.2f}")

if __name__ == "__main__":
    main()
