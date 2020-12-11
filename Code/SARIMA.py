import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from scipy import signal, stats
import copy
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import acf, plot_pacf, plot_acf
from MyFunctions import ADF_Cal, differencing, Q_val_cal, ACF_plot, autocorrelation_cal, series_autocorrelation_cal, corr_cal, phi_cal, GPAC_cal, LME, one_step_ARMA, h_step_ARMA
from sklearn.model_selection import train_test_split
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")


#%% --------- Load data
df = pd.read_csv("Preprocessed_AirQuality.csv", index_col="Date", parse_dates=True)
target = "NO2(GT)"

# splitting training and testing sets
df_train, df_test = train_test_split(df, test_size=0.2, shuffle=False)
train = df_train[target]
test = df_test[target]

#%% ----------------- Seasonal differencing -----------------------
# at first, we take a seasonal differencing of the time series
y = train.to_numpy()

# the seasonal period is 24, so we take a differencing of lag 24
y = differencing(y, lag = 24)

# ADF test for seasonal differenced data
rolling_mean = []
rolling_var = []
for i in range(1,len(y)):
    rolling_mean.append(np.mean(y[:i]))
    rolling_var.append(np.var(y[:i]))

# ADF test
print("\nADF test for seasonal differenced data:")
ADF_Cal(y)
print("\nADF test for rolling mean:")
ADF_Cal(rolling_mean)
print("\nADF test for rolling variance:")
ADF_Cal(rolling_var)

#%% ------------------- 1st differencing -------------------------
# Since the time series is not stationary after seasonal differencing
# We apply an additional 1st differencing


y = differencing(y, lag = 1)


# ADF test for seasonal differenced data
rolling_mean = []
rolling_var = []
for i in range(1,len(y)):
    rolling_mean.append(np.mean(y[:i]))
    rolling_var.append(np.var(y[:i]))

# ADF test
print("\nADF test for the time series after seasonal differencing followed by a 1st differencing:")
ADF_Cal(y)
print("\nADF test for rolling mean:")
ADF_Cal(rolling_mean)
print("\nADF test for rolling variance:")
ADF_Cal(rolling_var)

# differencing terms: d = 1, D = 1
#%% -------------------------- ACF, PCAF plot ---------------------------------
plt.figure()
plot_acf(y, lags=50, title="ACF plot of differenced data")
plt.xlabel("Lag")
plt.ylabel("Magnitude")
plt.show()


plt.figure()
plot_pacf(y, lags=50, title="PACF plot of differenced data")
plt.xlabel("Lag")
plt.ylabel("Magnitude")
plt.show()

# seasonal terms:
# a spike at lag 24 of ACF plot but no other spikes
# exponential decay in the seasonal lags of the PACF at lag= 24, 48, ...
# P = 0, Q = 1

#%% ------------------------ GPAC table
lags = 20
# ACF of y(t)
acf = series_autocorrelation_cal(y, lags)

# retrieve Ry
Ry = acf.loc[acf["lag"] > -1]
Ry.set_index("lag", inplace=True)
Ry = Ry["autocorrelation"].to_numpy()

# print and plot GPAC table
table = GPAC_cal(Ry,8,8)

# according to GPAC table:
# possible non-seasonal AR order is 1
# possible non-seasonal MA order is 1
# p = 1, q = 1

#%% -------------------------- SARIMA model
# SARIMA (1,1,1) (0,1,1)24
my_non_seasonal_order = (1,1,1)
my_seasonal_order = (0,1,1,24)

model = SARIMAX(train, order = my_non_seasonal_order, seasonal_order = my_seasonal_order)
model_fit = model.fit()

#%% --------------- Forecasting
forecast = model_fit.forecast(len(test))

#%% --------------- Prediction
prediction = model_fit.predict(start = train.index[0], end = train.index[len(train)-1])

#%% ---------------- plot
# plot the prediction
plt.figure()
plt.plot(train[-100:], label ="Training data")
plt.plot(prediction[-100:], label ="Fitted values")
plt.plot(test[:200], label="Testing data")
plt.plot(forecast[:200], label="Forecasted values")
plt.xlabel("Date")
plt.xticks(rotation=30)
plt.ylabel(target)
plt.title("Prediction of NO2 concentration using SARIMA (1,1,1) (0,1,1)24")
plt.legend(loc='best')
plt.show()

#%% -------------- Stats
residual = np.array(train) - np.array(prediction) # residuals
error = np.array(test) - np.array(forecast) # forecasted errors

# mean of residuals, MSE, estimated variance
SSE_train = np.square(residual).sum()
MSE_train = np.square(residual).mean()
print("MSE of fitted values is: ", MSE_train)
print("Mean of residuals is: ", np.mean(residual))
est_var_train = np.sqrt(SSE_train/(len(train)-3))
print("The estimated variance of residuals is:", est_var_train )


SSE_test = np.square(error).sum()
MSE_test = np.square(error).mean()
print("MSE of testing set is: ", MSE_test)
print("Mean of forecasted errors is: ", np.mean(error))

est_var_test = np.sqrt(SSE_test/(len(test)-3))
print("The estimated variance of forecasted errors is:", est_var_test )

#%%

print("Variance of residuals is:", np.var(residual))
print("Variance of errors is:", np.var(error))