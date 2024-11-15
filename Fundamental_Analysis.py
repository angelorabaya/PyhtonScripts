import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json


class CryptoAnalyzer:
    def __init__(self, symbol='BTC', interval='1d'):
        self.symbol = symbol
        self.interval = interval
        self.base_url = "https://api.binance.com/api/v3"
        self.fear_greed_url = "https://api.alternative.me/fng/"

    def fetch_market_data(self):
        """Fetch historical market data from Binance"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=200)

        params = {
            'symbol': f'{self.symbol}USDT',
            'interval': self.interval,
            'startTime': int(start_time.timestamp() * 1000),
            'endTime': int(end_time.timestamp() * 1000)
        }

        response = requests.get(f"{self.base_url}/klines", params=params)
        data = response.json()

        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close',
                                         'volume', 'close_time', 'quote_volume', 'trades',
                                         'taker_buy_base', 'taker_buy_quote', 'ignored'])

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        return df

    def calculate_technical_indicators(self, df):
        """Calculate various technical indicators"""
        # Moving Averages
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        df['BB_middle'] = df['close'].rolling(window=20).mean()
        df['BB_upper'] = df['BB_middle'] + 2 * df['close'].rolling(window=20).std()
        df['BB_lower'] = df['BB_middle'] - 2 * df['close'].rolling(window=20).std()

        # Volume-based indicators
        df['Volume_SMA'] = df['volume'].rolling(window=20).mean()

        # Money Flow Index
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']

        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=14).sum()

        money_ratio = positive_flow / negative_flow
        df['MFI'] = 100 - (100 / (1 + money_ratio))

        return df

    def get_market_sentiment(self):
        """Get Fear & Greed Index"""
        try:
            response = requests.get(self.fear_greed_url)
            data = response.json()
            return int(data['data'][0]['value'])
        except:
            return None

    def analyze_trend(self, df):
        """Comprehensive trend analysis"""
        current_price = df['close'].iloc[-1]

        signals = {
            'moving_averages': self.analyze_moving_averages(df),
            'momentum': self.analyze_momentum(df),
            'volatility': self.analyze_volatility(df),
            'volume': self.analyze_volume(df)
        }

        bullish_signals = sum(1 for signal in signals.values() if signal['signal'] == 'BULLISH')
        bearish_signals = sum(1 for signal in signals.values() if signal['signal'] == 'BEARISH')

        sentiment = self.get_market_sentiment()

        trend_strength = ((bullish_signals - bearish_signals) / len(signals)) * 100

        if trend_strength > 25:
            trend = "STRONG BULLISH"
        elif trend_strength > 0:
            trend = "MODERATE BULLISH"
        elif trend_strength < -25:
            trend = "STRONG BEARISH"
        elif trend_strength < 0:
            trend = "MODERATE BEARISH"
        else:
            trend = "NEUTRAL"

        return {
            'symbol': self.symbol,
            'current_price': round(current_price, 2),
            'trend': trend,
            'trend_strength': round(trend_strength, 2),
            'technical_signals': signals,
            'market_sentiment': sentiment,
            'risk_level': self.calculate_risk_level(df),
            'price_predictions': self.calculate_price_predictions(df)
        }

    def analyze_moving_averages(self, df):
        """Analyze moving averages crossovers and trends"""
        current_sma20 = df['SMA_20'].iloc[-1]
        current_sma50 = df['SMA_50'].iloc[-1]
        current_ema20 = df['EMA_20'].iloc[-1]

        signal = 'BULLISH' if current_sma20 > current_sma50 else 'BEARISH'
        strength = abs(current_sma20 - current_sma50) / current_sma50 * 100

        return {
            'signal': signal,
            'strength': round(strength, 2),
            'sma20': round(current_sma20, 2),
            'sma50': round(current_sma50, 2),
            'ema20': round(current_ema20, 2)
        }

    def analyze_momentum(self, df):
        """Analyze momentum indicators"""
        current_rsi = df['RSI'].iloc[-1]
        current_macd = df['MACD'].iloc[-1]
        current_macd_signal = df['MACD_signal'].iloc[-1]

        signal = 'BULLISH' if (current_rsi > 50 and current_macd > current_macd_signal) else 'BEARISH'

        return {
            'signal': signal,
            'rsi': round(current_rsi, 2),
            'macd': round(current_macd, 2),
            'macd_signal': round(current_macd_signal, 2)
        }

    def analyze_volatility(self, df):
        """Analyze volatility indicators"""
        volatility = df['close'].pct_change().std() * 100
        bb_width = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        current_bb_width = bb_width.iloc[-1]

        signal = 'BULLISH' if current_bb_width < bb_width.mean() else 'BEARISH'

        return {
            'signal': signal,
            'volatility': round(volatility, 2),
            'bb_width': round(current_bb_width, 2)
        }

    def analyze_volume(self, df):
        """Analyze volume indicators"""
        current_mfi = df['MFI'].iloc[-1]
        volume_sma = df['Volume_SMA'].iloc[-1]
        current_volume = df['volume'].iloc[-1]

        signal = 'BULLISH' if (current_mfi > 50 and current_volume > volume_sma) else 'BEARISH'

        return {
            'signal': signal,
            'mfi': round(current_mfi, 2),
            'volume_ratio': round(current_volume / volume_sma, 2)
        }

    def calculate_risk_level(self, df):
        """Calculate overall risk level"""
        volatility = df['close'].pct_change().std() * 100
        rsi = df['RSI'].iloc[-1]

        risk_score = 0
        risk_score += volatility * 2
        risk_score += abs(50 - rsi)

        if risk_score < 20:
            return "LOW"
        elif risk_score < 40:
            return "MODERATE"
        else:
            return "HIGH"

    def calculate_price_predictions(self, df):
        """Calculate price predictions using various methods"""
        current_price = df['close'].iloc[-1]

        trend_prediction = current_price * (1 + df['close'].pct_change().mean() * 30)

        return {
            'trend_based_30d': round(trend_prediction, 2),
            'resistance': round(df['BB_upper'].iloc[-1], 2),
            'support': round(df['BB_lower'].iloc[-1], 2)
        }

    def get_analysis(self):
        """Main method to get complete analysis"""
        try:
            df = self.fetch_market_data()
            df = self.calculate_technical_indicators(df)
            analysis = self.analyze_trend(df)
            return analysis
        except Exception as e:
            return f"Error: {str(e)}"


# Example usage
if __name__ == "__main__":
    analyzer = CryptoAnalyzer('ETH')
    analysis = analyzer.get_analysis()
    print("\nDetailed Cryptocurrency Analysis:")
    print(json.dumps(analysis, indent=2))