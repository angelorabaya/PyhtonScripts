import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

def get_SVM(currency):
    # Load data
    data = pd.read_csv(currency)

    # Feature engineering: create labels for bullish (1) or bearish (0)
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)

    # Drop the last row with NaN target
    data.dropna(inplace=True)

    # Features and target
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    X = data[features]
    y = data['Target']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train SVM
    model = SVC(kernel='linear', random_state=42)
    model.fit(X_train_scaled, y_train)

    # Make predictions
    predictions = model.predict(X_test_scaled)

    # Evaluate the model
    #print(classification_report(y_test, predictions, target_names=['Bearish', 'Bullish']))

    return classification_report(y_test, predictions, target_names=['Bearish', 'Bullish'])
