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
import warnings
warnings.filterwarnings("ignore")


#%% --------- Load data
df = pd.read_csv("Preprocessed_AirQuality.csv", index_col="Date", parse_dates=True)
target = "NO2(GT)"

# splitting training and testing sets
df_train, df_test = train_test_split(df, test_size=0.2, shuffle=False)
train = df_train[target]
test = df_test[target]

#%%
y = train.to_numpy()
# check stationary of original data
rolling_mean = []
rolling_var = []
for i in range(1,len(y)):
    rolling_mean.append(np.mean(y[:i]))
    rolling_var.append(np.var(y[:i]))

# ADF test
print("\nADF test for original data:")
ADF_Cal(y)
print("\nADF test for rolling mean:")
ADF_Cal(rolling_mean)
print("\nADF test for rolling variance:")
ADF_Cal(rolling_var)


plt.figure()
plot_acf(y, lags=48, title="ACF plot of NO2 concentrations")
plt.xlabel("Lag")
plt.ylabel("Magnitude")
plt.show()

#%% 1st differencing
# 1st differencing
y1 = differencing(y, 1)
# check for stationary
rolling_mean = []
rolling_var = []
for i in range(1,len(y)):
    rolling_mean.append(np.mean(y1[:i]))
    rolling_var.append(np.var(y1[:i]))

# ADF test
print("\nADF test for transformed data:")
ADF_Cal(y)
print("\nADF test for rolling mean:")
ADF_Cal(rolling_mean)
print("\nADF test for rolling variance:")
ADF_Cal(rolling_var)

#%%
plt.figure()
plot_acf(y1, lags=10, title="ACF plot of 1st differencing time series")
plt.xlabel("Lag")
plt.ylabel("Magnitude")
plt.show()

#%%
plt.figure()
plt.figure()
plot_pacf(y1, lags=10, title="PACF plot of 1st differencing time series")
plt.xlabel("Lag")
plt.ylabel("Magnitude")
plt.show()

#%%
lags = 48
# ACF of y(t)
acf = series_autocorrelation_cal(y1, lags)

# retrieve Ry
Ry = acf.loc[acf["lag"] > -1]
Ry.set_index("lag", inplace=True)
Ry = Ry["autocorrelation"].to_numpy()

# print and plot GPAC table
table = GPAC_cal(Ry,8,8)

#%% -------
# ARMA (2,1)
# na = 2
# nb = 1

# ARMA (5,5)
# na = 5
# nb = 5

# ARMA (1,10
na = 1
nb = 1
a, b, running_SSE, var_e, cov_theta = LME(y1,na,nb,100)


# 1-step prediction
y_train_pred, e_train = one_step_ARMA(y1,a,b)

# transform back to original order
y_pred = np.zeros(len(y))
for i in range(2, len(y)):
    y_pred[i] = y[i-1] + y_train_pred[i-1]


#%%
res = np.array(train[2:]) - y_pred[2:]
title = "ACF plot of residuals"
T = len(train)
h = 48
ACF_plot(res, 20, title)
DOF = h - na - nb
alpha = 0.000001
Q_val = Q_val_cal(res,h,T)
critical_Q = stats.chi2.ppf(1-alpha, DOF)
print("Q-value = ", Q_val)
print("Critical Q-value = ", critical_Q)

#%% ------------------------------------ Confidence interval ----------------------------------------------
if na != 0:
    for i in range(1,na + 1):
        print("The confidence interval for parameter a{:} = [{:.4f} , {:.4f}]".format(i, a[i] - 2*np.sqrt(cov_theta[i-1,i-1]), a[i] + 2*np.sqrt(cov_theta[i-1,i-1]) ) )

if nb != 0:
    for i in range(1,nb + 1):
        print("The confidence interval for parameter b{:} = [{:.4f} , {:.4f}]".format(i, b[i] - 2*np.sqrt(cov_theta[i+na-1,i+na-1]), b[i] + 2*np.sqrt(cov_theta[i+na-1,i+na-1]) ) )


#%% ------------------------------- Zero/ Pole ------------------------------------------------------
root_b = np.roots(b)
root_a = np.roots(a)

if nb != 0:
    for i in range(nb):
        print("The roots of numerators are: ", np.real(root_b[i]) )
if na != 0:
    for i in range(na):
        print("The roots of denominators are: ", np.real(root_a[i]) )

#%% stats
SSE_train = np.square(res).sum()
MSE_train = np.square(res).mean()
print("MSE of fitted values using LME is: ", MSE_train)
print("Mean of residuals is: ", np.mean(res))

est_var_train = np.sqrt(SSE_train/(len(train)-na-nb))
print("The estimated variance of residuals is:", est_var_train )

#%% h-step prediction
y_test_pred = h_step_ARMA(y1, y_train_pred, len(test),a,b)

# convert back to original order
y_forecast = np.zeros(len(test))
y_forecast[0] = y[-1] + y_test_pred[0]
for i in range(1,len(test)):
    y_forecast[i] = y_forecast[i-1] + y_test_pred[i]
#%%
inds1 = train[-100:].index
inds2 = test[:200].index
plt.figure()
plt.plot(inds1, y[-100:], label = "True values")
plt.plot(inds1, y_pred[-100:], label = "Fitted values")
plt.plot(inds2, np.array(test)[:200], label = "True values")
plt.plot(inds2, y_forecast[:200], label = "Forecasted values")
plt.title("1-step and Multi-step prediction using ARMA ({:},{:}".format(na,nb))
plt.xticks(rotation=40)
plt.ylabel(target)
plt.xlabel("Date")
plt.legend(loc = "best")
plt.show()

#%%
error = np.array(test) - y_forecast

SSE_test = np.square(error).sum()
MSE_test = np.square(error).mean()
print("MSE of testing set using LME is: ", MSE_test)
print("Mean of forecasted errors is: ", np.mean(error))

est_var_test = np.sqrt(SSE_test/(len(test)-na-nb))
print("The estimated variance of forecasted errors is:", est_var_test )

print("Variance of residuals is:", np.var(res))
print("Variance of errors is:", np.var(error))


