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


#%% ------------------------------------------------- Load data ------------------------------------------------------------
df = pd.read_csv("../data/Preprocessed_AirQuality.csv", index_col="Date", parse_dates=True)
target = "NO2(GT)"

# splitting training and testing sets
df_train, df_test = train_test_split(df, test_size=0.2, shuffle=False)
train = df_train[target]
test = df_test[target]

#%% ----------------------------------------------- Seasonal differencing --------------------------------------------------
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

#%% --------------------------------------------------- 1st differencing ---------------------------------------------------
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
#%% ------------------------------------------------- ACF, PCAF plot ------------------------------------------------------
plt.figure()
plot_acf(y, lags=100, title="Figure 17. ACF plot of differenced data")
plt.xlabel("Lag")
plt.ylabel("Magnitude")
plt.show()


plt.figure()
plot_pacf(y, lags=100, title="Figure 18. PACF plot of differenced data")
plt.xlabel("Lag")
plt.ylabel("Magnitude")
plt.show()

# seasonal terms:
# a spike at lag 24 of ACF plot but no other spikes
# exponential decay in the seasonal lags of the PACF at lag= 24, 48, ...
P = 0
Q = 1

#%% ------------------------------------------------ GPAC table ------------------------------------------------------------
# use GPAC table to find non-seasonal terms
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
p = 1
q = 1

#%% --------------------------------------------------- SARIMA model ---------------------------------------------------------
# SARIMA (1,1,1) (0,1,1)24
my_non_seasonal_order = (1,1,1)
my_seasonal_order = (0,1,1,24)

model = SARIMAX(train, order = my_non_seasonal_order, seasonal_order = my_seasonal_order, measurement_error=True)
model_fit = model.fit()

#%% ------------------------------------------------------ Forecasting -----------------------------------------------------
forecast = model_fit.forecast(len(test))

#%% ------------------------------------------------------ Prediction --------------------------------------------------
prediction = model_fit.predict(start = train.index[0], end = train.index[len(train)-1])

#%% -------------------------------------------------- Plot predictions and forecasts --------------------------------------
# plot the prediction
plt.figure()
plt.plot(train[-100:], label ="Training data")
plt.plot(prediction[-100:], label ="Fitted values")
plt.plot(test[:200], label="Testing data")
plt.plot(forecast[:200], label="Forecasted values")
plt.xlabel("Date")
plt.xticks(rotation=30)
plt.ylabel(target)
plt.title("Figure 19. NO2 concentration forecasting using SARIMA(1,1,1)(0,1,1)24")
plt.legend(loc='best')
plt.show()

#%% ------------------------------------------ Evaluation metrics ----------------------------------------------------------
# training set
residual = np.array(train) - np.array(prediction) # residuals
SSE_train = np.square(residual).sum()
MSE_train = np.square(residual).mean()
est_var_train = SSE_train/(len(train)-p-q-P-Q)
print("MSE of fitted values is: ", MSE_train)
print("Mean of residuals is: ", np.mean(residual))
print("The estimated variance of residuals is:", est_var_train )
print("Variance of residuals is:", np.var(residual))

# testing set
error = np.array(test) - np.array(forecast) # forecasted errors
SSE_test = np.square(error).sum()
MSE_test = np.square(error).mean()
est_var_test = SSE_test/(len(test)-p-q-P-Q)
print("MSE of testing set is: ", MSE_test)
print("Mean of forecasted errors is: ", np.mean(error))
print("The estimated variance of forecasted errors is:", est_var_test )
print("Variance of errors is:", np.var(error))

#%% ---------------------------------- ACF plot for residuals ----------------------------------------------------------
# acf plot and Q value
h = 48
title = "Figure 20. ACF plot of residuals using SARIMA model"
ACF_plot(residual, 20, title)
Q_val = Q_val_cal(residual, h, len(test))
print("Q-value of residuals is: ", Q_val)
DOF = h - p - q - P - Q
alpha = 0.05
critical_Q = stats.chi2.ppf(1-alpha, DOF)
print("Critical Q-value = ", critical_Q)

#%% ----------------------------------- Summary of SARIMA --------------------------------------------------------------
print(model_fit.summary())