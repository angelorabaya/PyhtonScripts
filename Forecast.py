import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#from ARIMA import get_ARIMA
#from PMD_Arima import get_PDARIMA
from RNNetwork import get_RNN
from Ensemble2 import CryptoPredictor
from Prophet import get_Prophet
#from SARIMA import get_SARIMA
#from SVM import get_SVM
#from Exponential_Smoothing import get_EXSMOOTH
#from Support_Vector_Regression import get_SVR
#from Gradient_Boosting_Machines import get_GBM
from Calculate_StopLoss import calculate_stop_loss_rrr

CURRENCY = "BTCUSDT.csv"

#get_arima_trend = get_ARIMA(CURRENCY)
#get_arima_trend = get_PDARIMA(CURRENCY)
get_rnn_trend = get_RNN(CURRENCY)

ensemble = CryptoPredictor(CURRENCY)
ensemble.prepare_data()
ensemble.train_model()
ensemble = ensemble.predict_trend()
get_ensemble_trend = ensemble['trend']
get_ensemble_price = ensemble['predicted_price']

#get_sarima_trend = get_SARIMA(CURRENCY,get_arima_trend[1],get_arima_trend[2],get_arima_trend[3])
#get_exponential_smoothing = get_EXSMOOTH(CURRENCY)
#get_support_vector_regression = get_SVR(CURRENCY)
#get_gradient_boosting_machine = get_GBM(CURRENCY)

prophet = get_Prophet(CURRENCY)
#support_vector_machine = get_SVM(CURRENCY)

print()
print(f"CURRENCY: {CURRENCY}")
#print(f"ARIMA: {get_arima_trend[0]}")
#print(f"SARIMA: {get_sarima_trend}")
print(f"RNN: {get_rnn_trend}")
print(f"ENSEMBLE: {get_ensemble_trend}")
#print(f"E-SMOOTHING: {get_exponential_smoothing}")
#print(f"S-VECTOR: {get_support_vector_regression}")
#print(f"G-BOOSTING: {get_gradient_boosting_machine}")
#print()
print(f"PROPHET: {prophet[1]}")
#print()
#print(prophet[0])
#print()
#print("SUPPORT VECTOR MACHINE:")
#print(support_vector_machine)

print()
last_close_price = round(prophet[2],4)
take_profit_bear = round(prophet[0].iloc[0].iloc[2],4)
take_profit_bull = round(prophet[0].iloc[-1].iloc[3],4)

print(f"CURRENCY: {CURRENCY}")
if (get_rnn_trend, get_ensemble_trend, prophet[1]) == ("Bullish", "Bullish", "Bullish"):
    print("Overall Market: Bullish")
    print(f"Last Closing Price: {last_close_price}")
    print(f"Take Profit: {take_profit_bull}")
    print(f"Stop Loss: {round(calculate_stop_loss_rrr(last_close_price,take_profit_bull,3)['stop_loss'],4)}")
elif (get_rnn_trend, get_ensemble_trend, prophet[1]) == ("Bearish", "Bearish", "Bearish"):
    print("Overall Market: Bearish")
    print(f"Last Closing Price: {last_close_price}")
    print(f"Take Profit: {take_profit_bear}")
    print(f"Stop Loss: {round(calculate_stop_loss_rrr(last_close_price, take_profit_bear, 3)['stop_loss'],4)}")