import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('d:/ETHUSDT.csv')

# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Use the 'Date' as the index
df.set_index('Date', inplace=True)

# Create a new DataFrame for the model
df['Days'] = (df.index - df.index[0]).days  # Number of days since start

# Feature and target variable
X = df[['Days']]
y = df['Close']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Visualize the results
plt.figure(figsize=(14, 7))
plt.scatter(df.index, df['Close'], color='blue', label='Actual Prices', alpha=0.5)
plt.scatter(X_test.index, predictions, color='red', label='Predicted Prices', alpha=0.5)
plt.title('Bitcoin Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# Predict future prices: Let's assume we're predicting for the next 30 days
future_days = pd.DataFrame({'Days': np.arange(df['Days'].max() + 1, df['Days'].max() + 31)})
future_predictions = model.predict(future_days)

# Print future predictions
future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30)
future_price_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_predictions})
print(future_price_df)