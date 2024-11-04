import pandas as pd
from pmdarima import auto_arima

def get_Arima_Values(currency):
    # Load the data
    data = pd.read_csv(currency, parse_dates=['Date'], index_col='Date')

    # Ensure the 'Close' column is in numeric format
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')

    # Resample the data to daily frequency (if needed)
    data = data['Close'].resample('D').mean()

    # Drop any missing values
    data = data.dropna()

    # Fit auto-ARIMA model
    model = auto_arima(data, seasonal=False, stepwise=True, trace=True, suppress_warnings=True)

    # Display model summary
    #print(model.summary())

    # Get the best ARIMA model parameters
    best_order = model.order
    p, d, q = best_order

    # Print the best parameters
    #print(f"Best model parameters: p={p}, d={d}, q={q}")

    return p,d,q