from locale import currency

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

def get_GBM(currency):
    # Load data
    data = pd.read_csv(currency)

    # Convert 'Date' to datetime format and set it as the index for time series analysis
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Feature Engineering
    def create_features(df):
        df['Price_Change'] = df['Close'].pct_change()
        df['Volatility'] = df['High'] - df['Low']
        df['Avg_Volume'] = df['Volume'].rolling(window=5).mean()
        df['Rolling_Close_Max'] = df['Close'].rolling(window=5).max()
        df['Rolling_Close_Min'] = df['Close'].rolling(window=5).min()
        df['Day_of_Week'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Year'] = df.index.year
        df.dropna(inplace=True)
        return df

    data = create_features(data)

    # Selecting Features and Target Variable
    X = data[['Open', 'High', 'Low', 'Volume', 'Price_Change', 'Volatility', 'Avg_Volume', 'Rolling_Close_Max', 'Rolling_Close_Min', 'Day_of_Week', 'Month', 'Year']]
    y = data['Close'].shift(-1).dropna()  # Predict next closing price
    X = X.loc[y.index]  # Align X with y

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the GBM model
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

    # Fit the model
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Future Predictions (using the latest available data)
    latest_data = data.iloc[-1][['Open', 'High', 'Low', 'Volume', 'Price_Change', 'Volatility', 'Avg_Volume', 'Rolling_Close_Max', 'Rolling_Close_Min', 'Day_of_Week', 'Month', 'Year']].values.reshape(1, -1)
    future_price = model.predict(latest_data)

    # Get the latest closing price
    latest_closing_price = data['Close'].iloc[-1]

    # Compare predicted price with the latest closing price
    if future_price[0] > latest_closing_price:
        trend = "Bullish"
    else:
        trend = "Bearish"

    #print(f'Predicted Future Price G-Boost: {future_price[0]}')
    #print(f'Market Trend: {market_trend}')

    return trend