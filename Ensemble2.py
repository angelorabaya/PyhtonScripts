import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

class CryptoPredictor:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        # Sort the dataframe by date in ascending order
        self.df = self.df.sort_values('Date', ascending=True)
        self.df.set_index('Date', inplace=True)
        self.rf_model = None
        self.scaler = None

    def prepare_data(self):
        self.df['MA7'] = self.df['Close'].rolling(window=7).mean()
        self.df['MA21'] = self.df['Close'].rolling(window=21).mean()
        self.df['EMA12'] = self.df['Close'].ewm(span=12, adjust=False).mean()
        self.df['EMA26'] = self.df['Close'].ewm(span=26, adjust=False).mean()
        self.df['MACD'] = self.df['EMA12'] - self.df['EMA26']
        self.df['RSI'] = self._calculate_rsi()
        self.df['Volatility'] = self.df['Close'].pct_change().rolling(window=21).std()
        self.df.dropna(inplace=True)

    def _calculate_rsi(self, period=14):
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def train_model(self):
        features = ['MA7', 'MA21', 'EMA12', 'EMA26', 'MACD', 'RSI', 'Volatility', 'High', 'Low', 'Close', 'Volume']
        X = self.df[features]
        y = self.df['Close'].shift(-1)

        X = X[:-1]
        y = y[:-1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_model.fit(X_train_scaled, y_train)

    def predict_trend(self, forecast_days=7):
        features = ['MA7', 'MA21', 'EMA12', 'EMA26', 'MACD', 'RSI', 'Volatility', 'High', 'Low', 'Close', 'Volume']
        last_data = self.df[features].iloc[-forecast_days:].values
        last_data_scaled = self.scaler.transform(last_data)
        future_prices = self.rf_model.predict(last_data_scaled)

        current_price = self.df['Close'].iloc[-1]  # Now this will be the most recent price
        avg_future_price = np.mean(future_prices)
        trend = "Bullish" if avg_future_price > current_price else "Bearish"
        trend_strength = abs(avg_future_price - current_price) / current_price * 100

        # Calculate confidence based on technical indicators
        rsi = self.df['RSI'].iloc[-1]
        macd = self.df['MACD'].iloc[-1]
        ma_trend = self.df['MA7'].iloc[-1] > self.df['MA21'].iloc[-1]

        confidence_score = 0
        confidence_score += 1 if rsi > 50 else -1
        confidence_score += 1 if macd > 0 else -1
        confidence_score += 1 if ma_trend else -1
        confidence = (confidence_score + 3) / 6 * 100  # Normalize to 0-100%

        return {
            'current_price': current_price,
            'predicted_price': avg_future_price,
            'trend': trend,
            'trend_strength': trend_strength,
            'confidence': confidence
        }


# Usage
#predictor = CryptoPredictor('BTCUSDT.csv')
#predictor.prepare_data()
#predictor.train_model()
#prediction = predictor.predict_trend()

#print(f"Current Price: ${prediction['current_price']:,.2f}")
#print(f"Predicted Price: ${prediction['predicted_price']:,.2f}")
#print(f"Trend: {prediction['trend']}")
#print(f"Trend Strength: {prediction['trend_strength']:.2f}%")
#print(f"Prediction Confidence: {prediction['confidence']:.2f}%")