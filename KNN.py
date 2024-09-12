import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the CSV data
data = pd.read_csv("d:/BTCUSDT.csv")

# Preprocess the data
data['Date'] = pd.to_datetime(data['Date'])
data['Target'] = (data['Close'].shift(-1) - data['Close']).apply(lambda x: 1 if x > 0 else 0)

# Drop rows with NaN values (last row after shifting)
data.dropna(inplace=True)

# Features and target variable
features = data[['Open', 'High', 'Low', 'Close', 'Volume']]
target = data['Target']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Function to get closest trade
def predict_close_trade(open_price, high_price, low_price, close_price, volume):
    input_data = scaler.transform([[open_price, high_price, low_price, close_price, volume]])
    prediction = knn.predict(input_data)
    return "Bullish" if prediction[0] == 1 else "Bearish"

# Example usage
if __name__ == "__main__":
    # Replace these values with your specific input
    open_price = float(input("Enter Open Price: "))
    high_price = float(input("Enter High Price: "))
    low_price = float(input("Enter Low Price: "))
    close_price = float(input("Enter Close Price: "))
    volume = float(input("Enter Volume: "))

    result = predict_close_trade(open_price, high_price, low_price, close_price, volume)
    print(f"The closest trade is predicted to be: {result}")