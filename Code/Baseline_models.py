import random
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.holtwinters as ets
import scipy
import numpy as np
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from sklearn.model_selection import train_test_split
from MyFunctions import simple_forecast_ts, forecasting_plot, stats, Q_val_cal, ACF_error, ACF_plot, autocorrelation_cal, series_autocorrelation_cal, corr_cal
import warnings
warnings.filterwarnings("ignore")

#%% ------------------------------------------- Set-up ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
#%% --------------------------------- Load data and split training and testing sets ---------------------------------------
df = pd.read_csv("../data/Preprocessed_AirQuality.csv", index_col="Date", parse_dates=True)
target = "NO2(GT)"  # target variable

# splitting training and testing sets
df_train, df_test = train_test_split(df, test_size=0.2, shuffle=False)
train = df_train[target]
test = df_test[target]

#%% ----------------------------------------- Baseline models & Holt-Winter ---------------------------------------------------------------

# plot training, testing sets and the forecasts
forecasting_plot(train, test, "Average", "Date", target, 2, "%Y-%m-%d", 0)
forecasting_plot(train, test, "Naive", "Date", target, 2, "%Y-%m-%d",  0)
forecasting_plot(train, test, "Drift", "Date", target, 2, "%Y-%m-%d",  0)
forecasting_plot(train, test, "Simple Exponential Smoothing", "Date", target ,2, "%Y-%m-%d", 0)
forecasting_plot(train, test, "Holt's Linear", "Date", target, 2, "%Y-%m-%d", 0)
forecasting_plot(train, test, "Holt-Winter", "Date", target, 2, "%Y-%m-%d", 24)

#%% ------------------------------------------ Evaluation metrics -----------------------------------------------------------
# MSE, variances of prediction and forecast errors, Q value and correlation coefficient
stats(train, test, "Average",0)
stats(train, test, "Naive",0)
stats(train, test, "Drift",0)
stats(train, test, "Simple Exponential Smoothing",0)
stats(train, test, "Holt's Linear",0)
stats(train, test, "Holt-Winter",24)

#%% ---------------------------------------------- ACF plots for residuals ----------------------------------------------
# ACF plots
ACF_error("Average", train, test,0)
ACF_error("Naive", train, test,0)
ACF_error("Drift", train, test,0)
ACF_error("Simple Exponential Smoothing", train, test,0)
ACF_error("Holt's Linear", train, test,0)
ACF_error("Holt-Winter", train, test,24)

#%% ------------------------------------ Critical Q-value --------------------------------------------------------------------
DOF = 48
alpha = 0.05
critical_Q = scipy.stats.chi2.ppf(1-alpha, DOF)
print("Critical Q-value = ", critical_Q)