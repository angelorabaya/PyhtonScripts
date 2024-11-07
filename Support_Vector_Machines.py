import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('DOGEUSDT.csv')

# Preprocess data
data['Date'] = pd.to_datetime(data['Date'])  # Convert date column to datetime
data['Date'] = data['Date'].map(pd.Timestamp.timestamp)  # Convert to seconds since epoch

# Define features and labels
features = data[['Date', 'Open', 'High', 'Low', 'Volume']]
labels = data['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the SVM model
svm_model = SVR(kernel='rbf')
svm_model.fit(X_train_scaled, y_train)

# Make predictions
predictions = svm_model.predict(X_test_scaled)

# Display predictions
predicted_df = pd.DataFrame({'Predicted': predictions, 'Actual': y_test.reset_index(drop=True)})
print(predicted_df)

# Optionally, save predictions to a new CSV
predicted_df.to_csv('predictions.csv', index=False)