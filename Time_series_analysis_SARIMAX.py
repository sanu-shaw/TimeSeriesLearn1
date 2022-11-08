import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('perrin-freres-monthly-champagne-.csv')

df.info()

df.isnull().sum()
df.dropna(inplace=True)

#cleaning up the data
df.columns=['Month','Sales']
df.set_index('Month', inplace=True)

df.describe()

plt.plot(df['Sales'].tail(50))
plt.show()

##Testing for stationarity

from statsmodels.tsa.stattools import adfuller

def adfuller_test(sales):
    result=adfuller(sales)
    print( "p-value "+str(result[1]))

adfuller_test(df['Sales'])


df['First diff'] = pd.Series.diff(df.Sales,periods=1)
df['seasonal diff']= pd.Series.diff(df['First diff'],periods=12)
plt.plot(df['seasonal diff'])
plt.show()

adfuller_test(df['seasonal diff'].dropna())  #p-value 0.0002650462849293356 hence rejecting the NUll Hypothesis and concluding the time seriese is Stationary


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = plot_acf(df['seasonal diff'].iloc[13:],lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = plot_pacf(df['seasonal diff'].iloc[13:],lags=40,ax=ax2)
plt.show()

##From ACF and PACF graph (p=2, d=1, q=1) ans S(p=1,d=12,q=1)
import statsmodels.api as sm

#model= sm.tsa.statespace.SARIMAX(df.Sales,order=(2,1,1), seasonal_order=(1,1,1,12))
results=model.fit()
results.summary()

pred = results.get_prediction(start=90,end=115,dynamic=False)

ax=df.Sales.plot(label='Observed')
pred.predicted_mean.plot(ax=ax,label='Forecast',alphs=.7)

