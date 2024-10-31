import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the data from CSV
data = pd.read_csv('BTCUSDT.csv')

# Convert the 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Sort data by date in ascending order
data = data.sort_values('Date')

# Prepare the features and target variable
data['Days'] = (data['Date'] - data['Date'].min()).dt.days  # Convert dates to numeric values
X = data[['Days']]
y = data['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions for the test set
predictions = model.predict(X_test)

# Calculate and print the mean squared error
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Prepare to predict the next 7 days
last_date_days = data['Days'].max()
future_days = np.array([[last_date_days + i] for i in range(1, 8)])  # Next 7 days

# Predict future prices
future_prices = model.predict(future_days)

# Convert numeric days back to dates correctly
future_dates_real = data['Date'].min() + pd.to_timedelta(future_days.flatten(), unit='D')  # Directly use Timedelta

# Display the future prices
future_prices_df = pd.DataFrame({'Date': future_dates_real, 'Predicted Close': future_prices})
print(future_prices_df)