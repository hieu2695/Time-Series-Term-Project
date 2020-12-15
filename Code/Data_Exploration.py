import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import acf, plot_pacf, plot_acf
from MyFunctions import ADF_Cal, ACF_plot, autocorrelation_cal, series_autocorrelation_cal, ts_strength
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

#%% ------------------------------------ Load dataset -------------------------------------------------------------------
# read the preprocessed dataset
df = pd.read_csv("../data/Preprocessed_AirQuality.csv", index_col="Date", parse_dates=True)

# get the target variable
target = "NO2(GT)"

# splitting into training and testing sets
df_train, df_test = train_test_split(df, test_size=0.2, shuffle=False)

# getting the training time series
ts = df_train[target]


#%% ----------------------------------------- Visualization of the time series -----------------------------------------------
# plot first 300 samples
fig, ax = plt.subplots()
ax.plot(ts[:300])
ax.xaxis.set_tick_params(reset=True)
ax.xaxis.set_major_locator(mdates.HourLocator(interval=48))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b/%d-%H"))
plt.setp(ax.get_xticklabels(), rotation=30, fontsize = 10)
plt.title("Figure 1. Hourly averaged NO2 concentration of 300 samples")
plt.xlabel("Date")
plt.ylabel(target + " concentration in microg/m^3")
plt.show()

#%% ------------------------------------------ ACF/PACF plot -----------------------------------------------------------
y = ts.to_numpy() # convert to np.array

# ACF
plt.figure()
plot_acf(y, lags=48, title="Figure 2. ACF plot of the time series")
plt.xlabel("Lag")
plt.ylabel("Magnitude")
plt.show()

# PACF
plt.figure()
plt.figure()
plot_pacf(y, lags=48, title="PACF plot of the time series")
plt.xlabel("Lag")
plt.ylabel("Magnitude")
plt.show()

#%% ------------------------------------------ ADF test for stationarity ----------------------------------------------------------------
# calculation of rolling mean and rolling variance
rolling_mean = []
rolling_var = []
for i in range(1,len(y)):
    rolling_mean.append(np.mean(y[:i]))
    rolling_var.append(np.var(y[:i]))

# ADF tests for the time series, rolling mean and rolling variance
print("\nADF test for the time series:")
ADF_Cal(y)
print("\nADF test for rolling mean:")
ADF_Cal(rolling_mean)
print("\nADF test for rolling variance:")
ADF_Cal(rolling_var)

#%% ------------------------------ Analysis Time series components ---------------------------------------------------------------------
# STL decomposition
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
plt.suptitle("Figure 3. STL decomposition (showing 1000 samples)")
plt.xlabel("Date")
plt.show()

# plot time series component seperately in a graph
Tt = res.trend  ## trend-cycle component
St = res.seasonal ## seasonal component
Rt = res.resid ## remainder component

plt.figure()
plt.plot(Tt[:500], label="Trend")
plt.plot(St[:500], label="Seasonality")
plt.plot(Rt[:500], label="Reminder")
plt.title("Time series components for 1000 samples")
plt.xticks(rotation = 30)
plt.xlabel("Date")
plt.ylabel("Magnitude")
plt.legend()
plt.show()



#%% ------------------------------ Strength of Trend-cycle and Seasonality ---------------------------------------------
# calculate the strength of trend-cycle and seasonaliry
Ft, Fs = ts_strength(St,Rt,Tt)

# print results
print("\n")
print("The strength of trend-cycle is: {:.4f}".format(Ft))
print("The strength of seasonality is: {:.4f}".format(Fs))


