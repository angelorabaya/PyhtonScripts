import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# Data loading and preprocessing
data = pd.read_csv('BTCUSDT.csv')
data['Returns'] = data['Close'].pct_change()
data['Signal'] = (data['Returns'] > 0).astype(int)

# Feature Engineering
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['RSI'] = 100 - (100 / (1 + (data['Close'].diff(1).fillna(0).apply(lambda x: max(x, 0)).rolling(window=14).mean() /
                           data['Close'].diff(1).fillna(0).apply(lambda x: abs(min(x, 0))).rolling(window=14).mean())))
data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
data.dropna(inplace=True)

X = data[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'RSI', 'MACD']]
y = data['Signal']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Building the model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu', kernel_regularizer='l2'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu', kernel_regularizer='l2'))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(16, activation='relu', kernel_regularizer='l2'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Fit the model with validation split
model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2, verbose=1, callbacks=[early_stopping])

# Predictions
predictions = model.predict(X_test)
signals = (predictions > 0.5).astype(int)

# Classification report
print(classification_report(y_test, signals))