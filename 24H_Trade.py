import pandas as pd
import requests
from datetime import datetime, timedelta
import yfinance as yf
from pycoingecko import CoinGeckoAPI
import numpy as np
import ta
from Crypto_Liquidation import CryptoLiquidationTracker
from Exchange_Flow import BlockchainMetricsTracker
from Ethereum_Whale_Activity import get_ethereum_whale_activity

class CryptoAnalyzer:
    def __init__(self):
        # Initialize API clients
        self.cg = CoinGeckoAPI()

    def get_technical_indicators(self, symbol='BTC-USD', timeframe='1d', limit=200):
        try:
            # Fetch OHLCV data using yfinance
            end_date = datetime.now()
            start_date = end_date - timedelta(days=limit)
            df = yf.download(symbol, start=start_date, end=end_date, interval=timeframe)

            # Custom RSI calculation
            def calculate_rsi(data, periods=14):
                delta = data.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
                rs = gain / loss
                return 100 - (100 / (1 + rs))

            # Custom MACD calculation
            def calculate_macd(data, fast=12, slow=26, signal=9):
                exp1 = data.ewm(span=fast, adjust=False).mean()
                exp2 = data.ewm(span=slow, adjust=False).mean()
                macd = exp1 - exp2
                signal_line = macd.ewm(span=signal, adjust=False).mean()
                return macd, signal_line

            # Custom Bollinger Bands calculation
            def calculate_bollinger_bands(data, window=20, std_dev=2):
                sma = data.rolling(window=window).mean()
                rolling_std = data.rolling(window=window).std()
                upper_band = sma + (rolling_std * std_dev)
                lower_band = sma - (rolling_std * std_dev)
                return upper_band, lower_band, sma

            # Custom Stochastic Oscillator calculation
            def calculate_stochastic(high, low, close, k_period=14, d_period=3):
                lowest_low = low.rolling(window=k_period).min()
                highest_high = high.rolling(window=k_period).max()
                k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
                d = k.rolling(window=d_period).mean()
                return k, d

            # Custom ATR calculation
            def calculate_atr(high, low, close, period=14):
                tr1 = high - low
                tr2 = abs(high - close.shift())
                tr3 = abs(low - close.shift())
                tr = pd.DataFrame({'TR1': tr1, 'TR2': tr2, 'TR3': tr3}).max(axis=1)
                atr = tr.rolling(window=period).mean()
                return atr

            # Calculate indicators using custom functions
            df['RSI'] = calculate_rsi(df['Close'])
            df['MACD'], df['MACD_signal'] = calculate_macd(df['Close'])
            df['BB_high'], df['BB_low'], df['BB_middle'] = calculate_bollinger_bands(df['Close'])
            df['Stoch_k'], df['Stoch_d'] = calculate_stochastic(df['High'], df['Low'], df['Close'])
            df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'])

            # Simple Moving Averages
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()

            # Add Volume Moving Average
            df['Volume_MA'] = df['Volume'].rolling(window=20).mean()

            # Add additional momentum indicators
            df['Price_Change'] = df['Close'].pct_change()
            df['ROC'] = df['Close'].pct_change(periods=12) * 100  # Rate of Change

            return df

        except Exception as e:
            print(f"Error calculating technical indicators: {str(e)}")
            return None

    def get_onchain_metrics(self, coin_id='bitcoin'):
        try:
            # Fetch on-chain data from CoinGecko
            data = self.cg.get_coin_by_id(coin_id)

            metrics = {
                'market_data': {
                    'current_price_usd': data['market_data']['current_price']['usd'],
                    'market_cap': data['market_data']['market_cap']['usd'],
                    'volume': data['market_data']['total_volume']['usd'],
                    'price_change_24h': data['market_data']['price_change_percentage_24h'],
                    'price_change_7d': data['market_data']['price_change_percentage_7d'],
                    'price_change_30d': data['market_data']['price_change_percentage_30d']
                },
                'developer_data': {
                    'forks': data['developer_data']['forks'],
                    'stars': data['developer_data']['stars'],
                    'subscribers': data['developer_data']['subscribers'],
                    'total_issues': data['developer_data']['total_issues'],
                    'closed_issues': data['developer_data']['closed_issues']
                    #'contributors': data['developer_data']['contributors']
                },
                'community_data': {
                    'twitter_followers': data['community_data']['twitter_followers'],
                    'reddit_subscribers': data['community_data']['reddit_subscribers'],
                    'reddit_active_accounts': data['community_data']['reddit_accounts_active_48h']
                }
            }

            return metrics

        except Exception as e:
            print(f"Error in onchain metrics: {str(e)}")
            return None

    def get_market_sentiment(self):
        try:
            # Get Fear and Greed Index
            fear_greed_url = "https://api.alternative.me/fng/"
            response = requests.get(fear_greed_url)
            fear_greed = response.json()['data'][0]

            return {
                'fear_greed_index': {
                    'value': fear_greed['value'],
                    'value_classification': fear_greed['value_classification'],
                    'timestamp': fear_greed['timestamp']
                }
            }

        except Exception as e:
            print(f"Error in market sentiment: {str(e)}")
            return None

    def get_whale_activity(self, address='1P5ZEDWTKTFGxQjZphgWPQUpe554WKDfHQ'):
        try:
            # Example whale address monitoring (using Blockchain.com API)
            url = f"https://blockchain.info/rawaddr/{address}"
            response = requests.get(url)
            data = response.json()

            return {
                'final_balance': data['final_balance'] / 100000000,  # Convert satoshi to BTC
                'total_received': data['total_received'] / 100000000,
                'total_sent': data['total_sent'] / 100000000,
                'n_tx': data['n_tx']
            }

        except Exception as e:
            print(f"Error in whale activity: {str(e)}")
            return None

    def get_derivatives_data(self, currency):
        try:
            # Using Binance Futures public endpoints
            binance_futures_url = "https://fapi.binance.com/fapi/v1"

            # Get Open Interest
            oi_response = requests.get(f"{binance_futures_url}/openInterest", params={'symbol': currency})
            oi_data = oi_response.json()

            # Get Funding Rate
            funding_response = requests.get(f"{binance_futures_url}/fundingRate",
                                            params={'symbol': currency, 'limit': 1})
            funding_data = funding_response.json()[0]

            return {
                'open_interest': float(oi_data['openInterest']),
                'funding_rate': float(funding_data['fundingRate']),
                'funding_time': funding_data['fundingTime']
            }

        except Exception as e:
            print(f"Error in derivatives data: {str(e)}")
            return None

    def get_correlation_data(self, currency):
        try:
            # Get correlation with major assets
            assets = [currency, 'SPY', 'GLD', 'UUP']
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)

            df = pd.DataFrame()

            for asset in assets:
                data = yf.download(asset, start=start_date, end=end_date)
                df[asset] = data['Close']

            correlation_matrix = df.corr()

            return correlation_matrix.to_dict()

        except Exception as e:
            print(f"Error in correlation data: {str(e)}")
            return None

    def get_additional_metrics(self, symbol='BTC-USD', timeframe='1d', limit=200):
        try:
            df = self.get_technical_indicators(symbol, timeframe, limit)

            # Volume-based indicators
            def calculate_obv(close, volume):
                return (np.sign(close.diff()) * volume).cumsum()

            def calculate_vwap(df):
                v = df['Volume']
                tp = (df['High'] + df['Low'] + df['Close']) / 3
                return (tp * v).cumsum() / v.cumsum()

            # Add Fibonacci Retracement levels
            def calculate_fibonacci_levels(high, low):
                diff = high - low
                levels = {
                    'fib_236': high - (diff * 0.236),
                    'fib_382': high - (diff * 0.382),
                    'fib_618': high - (diff * 0.618)
                }
                return levels

            # Market Profile metrics
            def calculate_poc(df, bins=50):
                return df['Close'].mode()[0]  # Simple POC calculation

            # Volatility indicators
            def calculate_keltner_channels(df, period=20, atr_multiplier=2):
                typical_price = (df['High'] + df['Low'] + df['Close']) / 3
                mean = typical_price.rolling(window=period).mean()
                atr = calculate_atr(df['High'], df['Low'], df['Close'], period)
                upper = mean + (atr * atr_multiplier)
                lower = mean - (atr * atr_multiplier)
                return upper, lower

            def calculate_atr(high, low, close, period=14):
                tr1 = high - low
                tr2 = abs(high - close.shift())
                tr3 = abs(low - close.shift())
                tr = pd.DataFrame({'TR1': tr1, 'TR2': tr2, 'TR3': tr3}).max(axis=1)
                atr = tr.rolling(window=period).mean()
                return atr

            indicators = {}

            # Calculate OBV, VWAP, and Keltner Channels
            indicators['OBV'] = calculate_obv(df['Close'], df['Volume'])
            indicators['VWAP'] = calculate_vwap(df)
            kc_upper, kc_lower = calculate_keltner_channels(df)
            indicators['KC_Upper'] = kc_upper
            indicators['KC_Lower'] = kc_lower

            # Calculate momentum and trend strength indicators
            adx_indicator = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'])
            indicators['ADX'] = adx_indicator.adx()
            indicators['DMP'] = adx_indicator.adx_pos()
            indicators['DMN'] = adx_indicator.adx_neg()

            # Calculate support/resistance levels
            indicators['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
            indicators['R1'] = 2 * indicators['Pivot'] - df['Low']
            indicators['S1'] = 2 * indicators['Pivot'] - df['High']

            # Calculate market microstructure metrics
            indicators['Price_Range'] = df['High'] - df['Low']
            indicators['Volume_Impact'] = indicators['Price_Range'] * df['Volume']

            # Calculate order flow indicators
            indicators['Delta'] = df['Volume'] * (df['Close'] > df['Open']).astype(int)
            indicators['CVD'] = indicators['Delta'].cumsum()  # Cumulative Volume Delta

            # Get the latest values for each indicator
            latest_indicators = {
                key: float(value.iloc[-1]) if hasattr(value, 'iloc') else float(value[-1])
                for key, value in indicators.items()
            }

            return latest_indicators

        except Exception as e:
            print(f"Error calculating additional metrics: {str(e)}")
            return None

    def get_market_depth(self, symbol='BTCUSDT'):
        try:
            # Get order book data from multiple exchanges
            binance_depth = requests.get(f"https://api.binance.com/api/v3/depth",
                                         params={'symbol': symbol, 'limit': 1000})

            # Calculate order book imbalance
            bids = pd.DataFrame(binance_depth.json()['bids'], columns=['price', 'quantity'], dtype=float)
            asks = pd.DataFrame(binance_depth.json()['asks'], columns=['price', 'quantity'], dtype=float)

            bid_sum = bids['quantity'].sum()
            ask_sum = asks['quantity'].sum()

            return {
                'bid_ask_ratio': bid_sum / ask_sum,
                'bid_wall_levels': bids.groupby('price')['quantity'].sum().nlargest(5).to_dict(),
                'ask_wall_levels': asks.groupby('price')['quantity'].sum().nlargest(5).to_dict()
            }

        except Exception as e:
            print(f"Error in market depth: {str(e)}")
            return None

    def get_all_metrics(self, currencya, currencyb, currencyc):
        metrics = {
            'technical_indicators': None,
            'onchain_metrics': None,
            'market_sentiment': None,
            'whale_activity': None,
            'derivatives_data': None,
            'correlation_data': None
        }

        # Collect all metrics
        technical = self.get_technical_indicators(symbol=currencyb)

        print(f"correlation_data: {self.get_correlation_data(currencyb)}")
        print(f"technical_indicators: {technical.tail(1).to_dict()}")
        print(f"onchain_metrics: {self.get_onchain_metrics(coin_id=currencyc)}")

        if currencyc == "bitcoin":
            print(f"market_sentiment: {self.get_market_sentiment()}")
            print(f"whale_activity: {self.get_whale_activity()}")
        elif currencyc == "ethereum":
            print(f"whale_activity: {get_ethereum_whale_activity()}")

        print(f"derivatives_data: {self.get_derivatives_data(currencya)}")
        print(f"additional_metrics: {self.get_additional_metrics(symbol=currencyb)}")
        print(f"market_depth: {self.get_market_depth(symbol=currencya)}")

        tracker = CryptoLiquidationTracker()
        data = tracker.get_aggregated_liquidation_data(currencya)
        print(f"liquidation_data: {data}")

        tracker_bmt = BlockchainMetricsTracker()
        if currencyc == "ethereum":
            tracker_bmt.blockchair_base_url = "https://api.blockchair.com/ethereum"

        bmt_data = tracker_bmt.get_metrics_data()
        blockchain_metrics = bmt_data['blockchain_metrics']
        print(f"exchange_flow: {blockchain_metrics}")

        return metrics

if __name__ == "__main__":
    analyzer = CryptoAnalyzer()
    metrics = analyzer.get_all_metrics('BTCUSDT','BTC-USD', 'bitcoin')
    #metrics = analyzer.get_all_metrics('ETHUSDT','ETH-USD', 'ethereum')