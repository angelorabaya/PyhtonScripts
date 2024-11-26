import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Dict, List


class CryptoFundamentalAnalysis:
    def __init__(self):
        self.coingecko_api = "https://api.coingecko.com/api/v3"

    def get_market_data(self, crypto_id: str) -> Dict:
        """Get comprehensive market data for a cryptocurrency"""
        endpoint = f"{self.coingecko_api}/coins/{crypto_id}?localization=false&tickers=false&market_data=true&community_data=true&developer_data=true&sparkline=true"
        response = requests.get(endpoint)
        return response.json()

    def get_price_history(self, crypto_id: str) -> List:
        """Get price history for volatility calculation"""
        endpoint = f"{self.coingecko_api}/coins/{crypto_id}/market_chart?vs_currency=usd&days=7&interval=daily"
        response = requests.get(endpoint)
        data = response.json()
        return [price[1] for price in data['prices']]

    def analyze_token_economics(self, crypto_id: str) -> Dict:
        """Analyze token economics and supply metrics"""
        try:
            market_data = self.get_market_data(crypto_id)['market_data']

            token_metrics = {
                'circulating_supply': market_data.get('circulating_supply', 0),
                'total_supply': market_data.get('total_supply', 0),
                'max_supply': market_data.get('max_supply', 0),
                'supply_ratio': (market_data.get('circulating_supply', 0) /
                                 market_data.get('total_supply', 1) if market_data.get('total_supply') else 0),
            }

            # Calculate inflation rate
            if token_metrics['circulating_supply'] and token_metrics['total_supply']:
                remaining_supply = token_metrics['total_supply'] - token_metrics['circulating_supply']
                token_metrics['inflation_rate'] = (remaining_supply / token_metrics['circulating_supply']) * 100
            else:
                token_metrics['inflation_rate'] = 0

            return token_metrics
        except Exception as e:
            return f"Error analyzing token economics: {str(e)}"

    def get_development_activity(self, crypto_id: str) -> Dict:
        """Analyze development activity"""
        try:
            data = self.get_market_data(crypto_id)
            dev_data = data.get('developer_data', {})

            dev_metrics = {
                'github_commits': dev_data.get('commit_count_4_weeks', 0),
                'github_stars': dev_data.get('stars', 0),
                'github_contributors': dev_data.get('pull_request_contributors', 0),
                'development_score': data.get('developer_score', 0)
            }

            return dev_metrics
        except Exception as e:
            return f"Error getting development activity: {str(e)}"

    def analyze_network_health(self, crypto_id: str) -> Dict:
        """Analyze network health metrics"""
        try:
            data = self.get_market_data(crypto_id)
            market_data = data.get('market_data', {})

            network_metrics = {
                'market_dominance': data.get('market_cap_rank', 0),
                'trading_volume': market_data.get('total_volume', {}).get('usd', 0),
                'liquidity_score': data.get('liquidity_score', 0),
                'community_score': data.get('community_score', 0)
            }

            return network_metrics
        except Exception as e:
            return f"Error analyzing network health: {str(e)}"

    def calculate_risk_metrics(self, crypto_id: str) -> Dict:
        """Calculate various risk metrics"""
        try:
            data = self.get_market_data(crypto_id)
            market_data = data.get('market_data', {})
            price_history = self.get_price_history(crypto_id)

            # Calculate volatility
            volatility = np.std(price_history) / np.mean(price_history) if price_history else 0

            risk_metrics = {
                'volatility_7d': volatility,
                'market_cap_rank': data.get('market_cap_rank', 0),
                'ath_change_percentage': market_data.get('ath_change_percentage', {}).get('usd', 0),
                'atl_change_percentage': market_data.get('atl_change_percentage', {}).get('usd', 0)
            }

            return risk_metrics
        except Exception as e:
            return f"Error calculating risk metrics: {str(e)}"

    def get_market_sentiment(self, crypto_id: str) -> Dict:
        """Get market sentiment metrics"""
        try:
            data = self.get_market_data(crypto_id)

            sentiment = {
                'coingecko_score': data.get('coingecko_score', 0),
                'sentiment_votes_up_percentage': data.get('sentiment_votes_up_percentage', 0),
                'sentiment_votes_down_percentage': data.get('sentiment_votes_down_percentage', 0),
                'public_interest_score': data.get('public_interest_score', 0)
            }

            return sentiment
        except Exception as e:
            return f"Error getting market sentiment: {str(e)}"

    def get_fundamental_score(self, crypto_id: str) -> Dict:
        """Calculate overall fundamental score"""
        try:
            # Get all metrics
            token_economics = self.analyze_token_economics(crypto_id)
            network_health = self.analyze_network_health(crypto_id)
            development = self.get_development_activity(crypto_id)
            risk_metrics = self.calculate_risk_metrics(crypto_id)
            sentiment = self.get_market_sentiment(crypto_id)

            # Initialize scores dictionary
            scores = {}

            # Token Economics Score (0-100)
            if isinstance(token_economics, dict):
                supply_ratio = token_economics.get('supply_ratio', 0)
                scores['token_economics'] = min(100, supply_ratio * 100)

            # Network Health Score (0-100)
            if isinstance(network_health, dict):
                market_dom = network_health.get('market_dominance', 0)
                volume = network_health.get('trading_volume', 0)
                scores['network_health'] = min(100, (1 / market_dom * 50 if market_dom else 0) +
                                               (min(50, volume / 1000000000)))

            # Development Score (0-100)
            if isinstance(development, dict):
                scores['development'] = min(100, development.get('development_score', 0) * 10)

            # Risk Score (0-100)
            if isinstance(risk_metrics, dict):
                volatility = risk_metrics.get('volatility_7d', 1)
                scores['risk'] = max(0, 100 - (volatility * 100))

            # Sentiment Score (0-100)
            if isinstance(sentiment, dict):
                scores['sentiment'] = sentiment.get('coingecko_score', 0)

            # Calculate weighted average score
            weights = {
                'token_economics': 0.25,
                'network_health': 0.25,
                'development': 0.20,
                'risk': 0.15,
                'sentiment': 0.15
            }

            final_score = sum(scores.get(k, 0) * weights[k] for k in weights)

            return {
                'individual_scores': scores,
                'final_score': final_score,
                'recommendation': self.get_recommendation(final_score)
            }

        except Exception as e:
            return f"Error calculating fundamental score: {str(e)}"

    def get_recommendation(self, score: float) -> str:
        """Generate investment recommendation based on fundamental score"""
        if score >= 80:
            return "Strong Buy - Excellent fundamentals"
        elif score >= 65:
            return "Buy - Good fundamentals"
        elif score >= 50:
            return "Hold - Average fundamentals"
        elif score >= 35:
            return "Watch - Below average fundamentals"
        else:
            return "Avoid - Weak fundamentals"


def main():
    analyzer = CryptoFundamentalAnalysis()

    # List of cryptocurrencies to analyze
    cryptos = ['bitcoin', 'ethereum', 'binancecoin']

    for crypto_id in cryptos:
        print(f"\n{'=' * 50}")
        print(f"Cryptocurrency Fundamental Analysis for {crypto_id.upper()}")
        print(f"{'=' * 50}")

        print("\n1. Token Economics:")
        print(analyzer.analyze_token_economics(crypto_id))

        print("\n2. Network Health:")
        print(analyzer.analyze_network_health(crypto_id))

        print("\n3. Development Activity:")
        print(analyzer.get_development_activity(crypto_id))

        print("\n4. Risk Metrics:")
        print(analyzer.calculate_risk_metrics(crypto_id))

        print("\n5. Market Sentiment:")
        print(analyzer.get_market_sentiment(crypto_id))

        print("\n6. Overall Fundamental Analysis:")
        print(analyzer.get_fundamental_score(crypto_id))

        # Add delay to respect API rate limits
        time.sleep(1)


if __name__ == "__main__":
    main()