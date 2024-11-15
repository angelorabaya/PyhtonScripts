import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

def get_SVR(currency):
    # Load the CSV data
    data = pd.read_csv(currency)

    # Convert Date to numeric format if necessary
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date'] = data['Date'].map(pd.Timestamp.timestamp)

    # Ensure data is in ascending order by Date
    data = data.sort_values('Date')

    # Features and target variable
    X = data[['Date', 'Open', 'High', 'Low', 'Volume']]
    y = data['Close']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # Scale the features
    scalerX = StandardScaler()
    scalerY = StandardScaler()

    X_train_scaled = scalerX.fit_transform(X_train)
    X_test_scaled = scalerX.transform(X_test)
    y_train_scaled = scalerY.fit_transform(y_train.values.reshape(-1, 1)).flatten()

    # Create and fit the SVR model
    model = SVR(kernel='rbf')
    model.fit(X_train_scaled, y_train_scaled)

    # Make predictions
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scalerY.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    # Compare last predicted price with the last actual price
    last_actual = y_test.iloc[-1]
    last_predicted = y_pred[-1]

    # Sentiment analysis
    if last_predicted > last_actual:
        trend = "Bullish"
    else:
        trend = "Bearish"

    return trend