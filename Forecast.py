from ARIMA import get_ARIMA
from RNNetwork import get_RNN
from Ensemble2 import CryptoPredictor
from Prophet import get_Prophet
from SARIMA import get_SARIMA

CURRENCY = "ADAUSDT.csv"

get_arima_trend = get_ARIMA(CURRENCY)
get_rnn_trend = get_RNN(CURRENCY)

ensemble = CryptoPredictor(CURRENCY)
ensemble.prepare_data()
ensemble.train_model()
ensemble = ensemble.predict_trend()

get_ensemble_trend = ensemble['trend']
get_ensemble_price = ensemble['predicted_price']
get_sarima_trend = get_SARIMA(CURRENCY,get_arima_trend[1],get_arima_trend[2],get_arima_trend[3])

prophet = get_Prophet(CURRENCY)

print(f"CURRENCY: {CURRENCY}")
print(f"ARIMA: {get_arima_trend[0]}")
print(f"SARIMA: {get_sarima_trend}")
print(f"RNN: {get_rnn_trend}")
print(f"ENSEMBLE: {get_ensemble_trend}")
#print(f"ENSEMBLE PREDICTED PRICE: {get_ensemble_price:.2f}")
print("PROPHET:")
print(prophet)