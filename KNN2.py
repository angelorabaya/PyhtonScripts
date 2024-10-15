import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Load data
df = pd.read_csv("EURUSD.csv")

# Preprocess data
df['Date'] = pd.to_datetime(df['Date'])
df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)  # 1 for bullish, 0 for bearish
df = df.dropna()  # Remove rows with NaN values

# Features and target variable
features = df[['Open', 'High', 'Low', 'Close', 'Volume']]
target = df['Target']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize and train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predictions
predictions = knn.predict(X_test)

# Find the last known data point
#last_row = df.iloc[-1]
last_row = df.iloc[0]

# Output the closest trade
predicted_trade = knn.predict(last_row[['Open', 'High', 'Low', 'Close', 'Volume']].values.reshape(1, -1))[0]

# Display trade information
signal = "Bullish" if predicted_trade == 1 else "Bearish"
entry_price = last_row['Close']
stop_loss = entry_price * 0.98 if predicted_trade == 1 else entry_price * 1.02  # example stop loss
take_profit = entry_price * 1.02 if predicted_trade == 1 else entry_price * 0.98  # example take profit

print(f"Predicted Trade: {signal}")
print(f"Entry Price: {entry_price}")
print(f"Stop Loss: {stop_loss}")
print(f"Take Profit: {take_profit}")