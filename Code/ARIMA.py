import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import acf, plot_pacf, plot_acf
from MyFunctions import ADF_Cal, differencing, Q_val_cal, ACF_plot, autocorrelation_cal, series_autocorrelation_cal, corr_cal, phi_cal, GPAC_cal
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


#%%
df = pd.read_csv("Preprocessed_AirQuality.csv", index_col="Date", parse_dates=True)
target = "NO2(GT)"
ts = df[target]

stl = STL(ts[:500])
res = stl.fit()
plt.figure()
fig = res.plot()
fig.axes[0].set_xticks([], [])
fig.axes[1].set_xticks([], [])
fig.axes[2].set_xticks([], [])
fig.axes[3].xaxis.set_major_locator(mdates.HourLocator(interval=48))
fig.axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d-%H:00"))
plt.setp(fig.axes[3].get_xticklabels(), rotation=30, fontsize=7)
plt.suptitle("STL decomposition for 500 samples ")
plt.xlabel("Date")
plt.show()



# splitting training and testing sets
df_train, df_test = train_test_split(df, test_size=0.2, shuffle=False)
train = df_train[target]
test = df_test[target]

# decompose training data
stl = STL(train)
res = stl.fit()
plt.figure()
fig = res.plot()
fig.axes[0].set_xticks([], [])
fig.axes[1].set_xticks([], [])
fig.axes[2].set_xticks([], [])
fig.axes[3].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
fig.axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(fig.axes[3].get_xticklabels(), rotation=30, fontsize=7)
plt.suptitle("STL decomposition for training set")
plt.xlabel("Date")
plt.show()

Tt = res.trend
St = res.seasonal
Rt = res.resid

plt.figure()
plt.plot(Tt[:500], label="Trend")
plt.plot(St[:500], label="Seasonality")
plt.plot(Rt[:500], label="Reminder")
plt.title("Time series components for 500 samples")
plt.xticks(rotation = 30)
plt.xlabel("Date")
plt.ylabel("Magnitude")
plt.legend()
plt.show()

#%%------- Seasonal Adjusted data
At = train - St

plt.figure()
plt.plot(train[:500], label = "Original Data")
plt.plot(At[:500], label ="Adjusted Seasonal Data")
plt.title("Seasonal Adjustment")
plt.xticks(rotation = 30)
plt.xlabel("Date")
plt.ylabel("Magnitude")
plt.legend()
plt.show()

#%% Strength

def ts_strength(S, R, T):
    Ft = max([0, 1 - np.var(R)/np.var(T+R)])
    Fs = max([0, 1 - np.var(R)/np.var(S+R)])
    return Ft, Fs

Ft, Fs = ts_strength(St,Rt,Tt)
print("\n")
print("The strength of trend is: {:.4f}".format(Ft))
print("The strength of seasonality is: {:.4f}".format(Fs))

#%% stationarity

y = At.to_numpy()
# 1st differencing
y = differencing(y,lag=1)
#y = differencing(y,lag=1)

# check for stationary
rolling_mean = []
rolling_var = []
for i in range(1,len(y)):
    rolling_mean.append(np.mean(y[:i]))
    rolling_var.append(np.var(y[:i]))

# ADF test
print("\nADF test for transformed data:")
ADF_Cal(y)
print("\nADF test for rolling mean:")
ADF_Cal(rolling_mean)
print("\nADF test for rolling variance:")
ADF_Cal(rolling_var)


#%%-------------
plt.figure()
plot_acf(y, lags=20)
plt.show()

#%%
plt.figure()
plt.figure()
plot_pacf(y, lags=20)
plt.show()

#%%----
#%%
acf = series_autocorrelation_cal(y, lags=96)
# retrieve Ry
Ry = acf.loc[acf["lag"] > -1]
Ry.set_index("lag", inplace=True)
Ry = Ry["autocorrelation"].to_numpy()

# print and plot GPAC table
table = GPAC_cal(Ry,7,7)

#%% ARIMA(1,1,1) and ARIMA (3,1,3)

from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.arima.model import ARIMA

train.index.freq = train.index.inferred_freq
stlf = STLForecast(train, ARIMA, model_kwargs=dict(order=(1,1,1)))
stlf_res = stlf.fit()
forecast = stlf_res.forecast(len(test))
stlf_res.summary()


#%%
errors = np.array(test) - np.array(forecast)
SSE_test = np.square(errors).sum()
MSE_test = np.square(errors).mean()

print("MSE of forecasted values is: ", MSE_test)
r = corr_cal(np.array(test),np.array(forecast))
print(" The variance of forecasted errors is:", np.var(errors) )
print("Correlation coefficient between forecasted values and testing data is: ", r)
#%%
prediction = stlf_res.get_prediction(train.index[0],train.index[len(train)-1]).predicted_mean
residuals = np.array(train) - np.array(prediction)
SSE_train = np.square(residuals).sum()
MSE_train = np.square(residuals).mean()

print("MSE of fitted values is: ", MSE_train)
r = corr_cal(np.array(train),np.array(prediction))
print("Correlation coefficient between fitted values and training data is: ", r)
print("Mean of residuals is:" , np.mean(residuals))
title = "ACF plot of residuals using ARIMA (1,1,1)"
ACF_plot(residuals, 20, title)
Q_val = Q_val_cal(residuals, 48,len(test))
print("Q-value of residuals is: ", Q_val)
print(" The variance of residuals is:", np.var(residuals) )


#%%
na = stlf_res.model.k_ar
nb = stlf_res.model.k_ma
est_var_train = np.sqrt(SSE_train/(len(train)-na-nb))
print("The estimated variance of residuals is:", est_var_train )

est_var_test = np.sqrt(SSE_test/(len(test)-na-nb))
print("The estimated variance of forecasted errors is:", est_var_test )

#%%
# plot the prediction
plt.figure()
plt.plot(train[-100:], label ="Training data")
plt.plot(prediction[-100:], label ="Fitted values")
plt.plot(test[:200], label="Testing data")
plt.plot(forecast[:200], label="Forecasted values")
plt.xlabel("Date")
plt.xticks(rotation=30)
plt.ylabel(target)
plt.title("Forecasting of NO2 concentration using ARIMA (1,1,1)")
plt.legend(loc='best')
plt.show()