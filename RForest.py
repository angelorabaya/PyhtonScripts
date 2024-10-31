import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Load the data
data = pd.read_csv('BTC_USDT.csv', parse_dates=['Date'])
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Prepare features and target for the next 5 days
for i in range(1, 6):  # Shift Close column by 1 to 5 days
    data[f'Target_{i}'] = data['Close'].shift(-i)

# Drop rows with NaN values after shifting
data.dropna(inplace=True)

# Features and target
features = data[['Open', 'High', 'Low', 'Close', 'Volume']]
target = data[['Target_1', 'Target_2', 'Target_3', 'Target_4', 'Target_5']]  # targets for the next 5 days

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create the model
model = RandomForestRegressor(random_state=42)

# Hyperparameter tuning
param_distributions = {
    'n_estimators': np.arange(50, 300, 50),
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=20,  # Number of parameter settings to sample
    cv=5,  # Number of cross-validation folds
    scoring='neg_mean_squared_error',
    n_jobs=-1,  # Use all available cores
    random_state=42
)

random_search.fit(X_train, y_train)

# Make predictions for the next 5 days on the test set
predictions = random_search.predict(X_test)

# Create a DataFrame to display the predictions
predictions_df = pd.DataFrame(predictions, columns=['Target_1', 'Target_2', 'Target_3', 'Target_4', 'Target_5'])

# Print the predictions
print(predictions_df.head())  # Adjust the `head()` parameter to see more rows if needed

# If you want to make predictions for a specific recent date, for the last row of features:
latest_features = features.iloc[-1].values.reshape(1, -1)  # Reshape for prediction
latest_predictions = random_search.predict(latest_features)

# Create a DataFrame to display the predictions for the last row
latest_predictions_df = pd.DataFrame(latest_predictions, columns=['Target_1', 'Target_2', 'Target_3', 'Target_4', 'Target_5'])

# Print the latest predictions
print(latest_predictions_df)