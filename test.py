from yfinance import Ticker
import pandas as pd 

df = Ticker('AAPL').history(period='max')

# Data Format 
df.rename(columns={'Close':'Value'}, inplace=True)
df = df[['Value']]

# Simple Exponential Smoothing 

from statsmodels.tsa.holtwinters import ExponentialSmoothing


print(
        ExponentialSmoothing(df)

)
import statsmodels.api as sm

# Create a DataFrame with a time series data
data = {'Month': ['Jan', 'Feb', 'Mar', 'Apr'],
        'Sales': [100, 120, 110, 130]}
df = pd.DataFrame(data)
# Set the 'Month' column as the index
df.set_index('Month', inplace=True)
# Perform simple exponential smoothing
model = sm.tsa.SimpleExpSmoothing(df['Sales'])
ses_model = model.fit(smoothing_level=0.3)
df['Forecast'] = ses_model.fittedvalues
# Perform double exponential smoothing
model = sm.tsa.ExponentialSmoothing(df['Sales'], trend='add')
des_model = model.fit(smoothing_level=0.3, smoothing_slope=0.2)
df['Forecast'] = des_model.fittedvalues
# Perform triple exponential smoothing
model = sm.tsa.ExponentialSmoothing(df['Sales'], trend='add', seasonal='add', seasonal_periods=3)
tes_model = model.fit(smoothing_level=0.3, smoothing_slope=0.2, smoothing_seasonal=0.4)
df['Forecast'] = tes_model.fittedvalues
# Print the DataFrame with the forecasts
print(df)

