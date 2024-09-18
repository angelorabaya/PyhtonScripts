import pandas as pd
import matplotlib

df = pd.read_csv('BNBUSDT.csv', parse_dates=['Date'], index_col='Date')
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df = df['Close'].resample('D').mean()

df.info()
df.plot()