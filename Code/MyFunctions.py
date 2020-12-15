#%% --------------------------------------------- Import --------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.holtwinters as ets
import seaborn as sns
import numpy as np
from scipy import signal
import copy
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore")

# correlation coefficient
def corr_cal(x,y):
    """
    Parameters
    ----------
    x : data
    y : data with the same length as x

    Returns : the correlation coefficient between x and y
    -------

    """
    if len(x) != len(y):
        print("Two datasets do not have the same length. Recheck the datasets")
    else:
        length = len(x)
        # calculate means
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        # calculate the numerator of the correlation coefficient function
        numerator = 0
        for i in range(0,length):
            numerator = numerator + (x[i]-x_mean)*(y[i]-y_mean)

        # calculate the denominator of the correlation coefficient function
        X = 0
        Y = 0
        for i in range(0, length):
            X = X + (x[i]-x_mean)**2
            Y = Y + (y[i]-y_mean)**2
        denominator = (np.sqrt(X))*(np.sqrt(Y))

        # calculate the correlation coefficient
        r = numerator/denominator

        return r

# define a function to calculate the autocorrelation at lag k
def autocorrelation_cal(y,k):
    """
    calculate the autocorrelation between lagged values with distance k
    :param y: lagged values
    :param k: distance (k time units) that observations of a time series are separated
    :return: autocorrelation
    """
    # initiate the numerator and denominator for the autocorrelation
    numerator = 0
    denominator = 0
    meanY = np.mean(y)

    # calculate the numerator for autocorrelation
    for i in range(k,len(y)):
        numerator = numerator + (y[i]-meanY)*(y[i-k]-meanY)

    # calculate the denominator for autocorrelation
    for i in range(len(y)):
        denominator = denominator + (y[i]-meanY)**2
    # calculate the autocorrelation
    t = numerator/denominator

    return t

# define a function to calculate autocorrelations for all lags (two sides)
def series_autocorrelation_cal(y, lags):
    """
    calculate the autocorrelation between lagged values with every distance
    :param y: lagged values
    :return: a series of autocorrelations for the whole time domain
    """
    autocorrs = []
    nlags = []

    # we also consider the autocorrelation at (-k) which is equal to the auto correlation at k
    for i in range(-lags,lags+1):
        k = abs(i)
        autocorrs.append(autocorrelation_cal(y,k))
        nlags.append(i)

    df = pd.DataFrame({"lag":nlags,"autocorrelation":autocorrs})

    return df

# define a function to plot ACF
def ACF_plot(Y,lags,title):
    """
    # plot the ACF of white noise
    :param Y: the lagged values
    :param lags: number of lags
    :param title: title for the plot
    :return: ACF plot
    """
    df = series_autocorrelation_cal(Y, lags)
    # create values for x axis using the number of lags
    x = np.arange(-lags,lags + 1,1)
    # create a list to store values of y
    y = []
    for i in x:
        # get the autocorrelation at lag i
        ac = df.loc[df["lag"] == i]["autocorrelation"]
        # aapend the autocorrelation to y
        y.append(ac.values[0])
    # plot the ACF

    plt.figure(figsize=(7,7))
    plt.stem(x,y)
    plt.title(title, fontsize = 13)
    plt.xlabel("Lags", fontsize =10)
    plt.ylabel("Magnitude",fontsize = 10)
    plt.grid()
    plt.show()

# define a function to forcast the data using different methods
def simple_forecast_ts(train, test, method ,period):
    """
    :param ytrain: training data
    :param ytest:  testing data
    :param method: simple forecast method
    :param period: only use for Holt-Winter Additive method to define the seasonal_periods
    :return: dataframes containing prediction/forecast, errors and square errors
    """
    T = len(train)  # number of observations in the training set
    h = len(test)  # number of observations to be forecast4ed in the testing set

    ytrain = train.values
    ytest = test.values

    xtrain = train.index
    xtest = test.index

    ytrain_hat = [] # prediction
    ytest_hat = []  # forecast
    etrain = []  # prediction error
    etest = []  # forecast error
    setrain = []  # square of prediction error
    setest = []  # square of forecast error


    if method == 'Average': # using average method
        for i in range(0, 1):
            ytrain_hat.append(np.nan)
            etrain.append(np.nan)
            setrain.append(np.nan)
        for i in range(1,T):
            prediction = np.mean(ytrain[0:i])
            ytrain_hat.append(prediction)
            error = ytrain[i] - prediction
            etrain.append(error)
            setrain.append(error**2)

        for i in range(0,h):
            forecast = np.mean(ytrain)
            ytest_hat.append(forecast)
            error = ytest[i]-forecast
            etest.append(error)
            setest.append(error**2)


    elif method == 'Naive':  # naive method
        for i in range(0, 1):
            ytrain_hat.append(np.nan)
            etrain.append(np.nan)
            setrain.append(np.nan)
        for i in range(1, T):
            prediction = ytrain[i-1]
            ytrain_hat.append(prediction)
            error = ytrain[i]-prediction
            etrain.append(error)
            setrain.append(error**2)

        for i in range(0, h):
            forecast = ytrain[T-1]
            ytest_hat.append(forecast)
            error = ytest[i] - forecast
            etest.append(error)
            setest.append(error**2)

    elif method == 'Drift':  # drift method
        for i in range(0, 2):
            ytrain_hat.append(np.nan)
            etrain.append(np.nan)
            setrain.append(np.nan)
        for i in range(2, T):
            prediction = ytrain[i - 1] + (ytrain[i-1]-ytrain[0])/(i-1)
            ytrain_hat.append(prediction)
            error = ytrain[i] - prediction
            etrain.append(error)
            setrain.append(error**2)

        for i in range(0, h):
            forecast = ytrain[T - 1] + (ytrain[T-1]-ytrain[0])*(i+1)/(T-1)
            ytest_hat.append(forecast)
            error = ytest[i] - forecast
            etest.append(error)
            setest.append(error**2)

    elif method == "Simple Exponential Smoothing":  # simple exponential smoothing method
        alpha = 0.5
        ytrain_hat.append(ytrain[0]) # the first observation is the initial condition
        etrain.append(np.nan)  # for the initial condition, there is no error
        setrain.append(np.nan)

        for i in range(1, T):
            prediction = alpha*ytrain[i - 1] + (1-alpha)*ytrain_hat[i-1]
            ytrain_hat.append(prediction)
            error = ytrain[i] - prediction
            etrain.append(error)
            setrain.append(error**2)

        for i in range(0, h):
            forecast = alpha*ytrain[T - 1] + (1-alpha)*ytrain_hat[T-1]
            ytest_hat.append(forecast)
            error = ytest[i] - forecast
            etest.append(error)
            setest.append(error**2)

    elif method == "Holt's Linear Multiplicative": # Holt linear using multiplicative
        holt = ets.ExponentialSmoothing(ytrain, trend='multiplicative', damped=True,seasonal=None).fit()
        ytrain_hat = holt.fittedvalues
        ytest_hat = holt.forecast(steps=h)

        for i in range(0,T):
            error = ytrain[i]-ytrain_hat[i]
            etrain.append(error)
            setrain.append(error**2)
        for i in range(0,h):
            error = ytest[i] - ytest_hat[i]
            etest.append(error)
            setest.append(error**2)

    elif method == "Holt's Linear":
        holt = ets.ExponentialSmoothing(ytrain, trend=None, damped=False, seasonal=None).fit(smoothing_level=0.1)
        ytrain_hat = holt.fittedvalues
        ytest_hat = holt.forecast(steps=h)

        for i in range(0, T):
            error = ytrain[i] - ytrain_hat[i]
            etrain.append(error)
            setrain.append(error ** 2)
        for i in range(0, h):
            error = ytest[i] - ytest_hat[i]
            etest.append(error)
            setest.append(error**2)



    elif method == "Holt-Winter":
        holt = ets.ExponentialSmoothing(train, seasonal_periods=period, trend='add', damped_trend=True, seasonal='additive')
        holt = holt.fit(smoothing_level=0.1,smoothing_seasonal=0.2, smoothing_trend=None)
        ytrain_hat = holt.fittedvalues
        ytrain_hat = ytrain_hat.values
        ytest_hat = holt.forecast(steps=h)
        ytest_hat =ytest_hat.values


        for i in range(0, T):
            error = ytrain[i] - ytrain_hat[i]
            etrain.append(error)
            setrain.append(error**2)
        for i in range(0, h):
            error = ytest[i] - ytest_hat[i]
            etest.append(error)
            setest.append(error**2)

    else:
        print("Method is not applicable")

    # create dataframes containing results after predicting the training data and forecasting the testing data
    df_train = pd.DataFrame({"time": xtrain , "y_t": ytrain, "hat_y_t":ytrain_hat,
                             "error": etrain, "square_error":setrain})

    df_test = pd.DataFrame({"time": xtest, "y_t": ytest, "hat_y_t": ytest_hat,
                             "error": etest, "square_error": setest})

    return df_train, df_test

# Q value of Box-pierce test
def Q_val_cal(y, lags, T):  # calculating the Q-value in Box-Pierce test
    """
    :param y: residuals
    :param lags: number of lags
    :param T: size of training set
    :return: Q-value of Box-Pierce test
    """
    Q = 0
    for i in range(1, lags + 1):
        r = autocorrelation_cal(y, i)
        Q = Q + (r**2)*T

    return Q

# define a function to plot yearly data and the forecast values
def forecasting_plot(train,test, method, xlabel, ylabel, interval, dateformat, period):
    """
    :param train: training set
    :param test: testing set
    :param method: forecasting method
    :param xlabel: x axis label
    :param ylabel: y axis label
    :param interval: the difference between the ticks on the x axis
    :param dateformat: the format of tick labes on the x axis
    :param period: seasonal period for Holt-Winter additive method
    :return: plot of training, testing data and forecasts
    """
    prediction = simple_forecast_ts(train, test, method, period)[0]
    forecast = simple_forecast_ts(train, test, method, period)[1]

    # plot the training, testing sets and forecasting values
    fig, ax = plt.subplots()
    ax.plot(train[-100:], label = "Training data")
    ax.plot(train[-100:].index, prediction["hat_y_t"][-100:], label="Predictions")
    ax.plot(test[:200], label ="Testing data")
    ax.plot(test[:200].index, forecast["hat_y_t"][:200], label="Forecasts")

    ax.xaxis.set_tick_params(reset=True)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(dateformat))
    plt.setp(ax.get_xticklabels(), rotation=40, fontsize=10)
    plt.legend(loc="best")
    plt.title(method + " method forecasts", fontsize = 13)
    plt.xlabel(xlabel, fontsize = 13)
    plt.ylabel(ylabel, fontsize = 13)
    plt.show()


# calculate the MSE, variance, Q value and correlation coefficient
def stats(train, test, method, period):
    """
    :param train: training set
    :param test: testing set
    :param method: forecasting method
    :return: print MSE of forecasts, var of prediction, var of forecast, Q-value
            and correlation coefficient between forecast errors and testing data
    """
    prediction = simple_forecast_ts(train, test, method, period)[0]
    forecast = simple_forecast_ts(train, test, method, period)[1]

    # Mean square of errors
    MSE_test = np.mean(forecast["square_error"])
    MSE_train = np.mean(prediction["square_error"])

    # variance of prediction and forecast errors
    pred_var = np.var(prediction["error"])
    forecast_var = np.var(forecast["error"])


    # Q value
    lags = 48
    res = prediction["error"][3:]
    res = res.reset_index(drop=True)
    T = len(res)
    Q = Q_val_cal(res, lags, T)

    # correlation coefficient between forecast errors and the test set
    r = corr_cal(forecast["error"],test.values)

    print("\n")
    print("The mean of residuals is:", np.mean(res))
    print("The MSE of predictions using " + method + " is: {:.4f}".format(MSE_train))
    print("The MSE of forecasts using " + method + " is: {:.4f}".format(MSE_test))
    print("The variance of prediction errors using " + method + " is: {:.4f}".format(pred_var))
    print("The variance of forecast errors using " + method + " is: {:.4f}".format(forecast_var))
    print("The Q-value of residuals using " + method + " is: {:.4f}".format(Q))
    print("The correlation coefficient between forecast errors and the testing data using " + method + " is: {:.4f}".format(r))


# define a function to plot ACF for forecast errors
def ACF_error(method, train, test, period):
    prediction = simple_forecast_ts(train, test, method, period)[0]
    res = prediction["error"][3:]
    res = res.reset_index(drop=True)
    T = len(res)
    lags = 20
    ACF_plot(res, lags, "The ACF plot of residuals using " + method)


# define the ADF-test calculation
def ADF_Cal(x):
    result = adfuller(x)

    print('ADF Statistic: %f' %result[0])
    print('p_value: %f' %result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

# differencing (default 1st differencing
def differencing(y, lag=1):
	diff = np.zeros(len(y)-lag)
	for i in range(lag, len(y)):
		diff[i-lag] = y[i] - y[i - lag]
	return diff

# define a function to calculate phi(j,kk) component of GPAC table
def phi_cal(Ry, k, j):
    num = []
    den = []
    for i_r in np.arange(j, j + k):
        r_num = []
        r_den = []
        for i_c in np.arange(i_r + 1 - k, i_r + 1):
            index = abs(i_c)
            r_num.append(Ry[index])
            r_den.append(Ry[index])

        # reversing the list
        r_num = r_num[::-1]
        r_num[-1] = Ry[i_r + 1] # replace last value of r_num

        r_den = r_den[::-1]


        # appending to num and den
        num.append(r_num)
        den.append(r_den)


    # convert to matrix
    num = np.array(num)
    den = np.array(den)
    num = np.float64(np.linalg.det(num))
    den = np.linalg.det(den)

    if abs(den) > 1e-6:
        phi = float("{:.3f}".format(num/den))
    else:
        phi = float("inf")

    return phi

# define a function to calculate GPAC table
def GPAC_cal(Ry, K, J):

    phi = np.zeros(shape=(J+1,K))
    for k in range(1,K+1):
        for j in range(J+1):
            phi[j,k-1] = phi_cal(Ry,k,j)
    table  = pd.DataFrame(data=phi,
                          index=[i for i in range(J+1)],
                          columns=[i for i in range(1,K+1)])


    ax = sns.heatmap(table, vmin=-1, vmax=1, center=0,
                     cmap=sns.diverging_palette(20, 220, n=200),
                     annot=True,fmt=".3f")
    ax.set_title("Generalized Partial Autocorrelation (GPAC) Table")
    plt.xlabel("k")
    plt.ylabel("j")
    plt.show()

    print("\nGPAC table:")
    print(table)

# Levenberg-Marquardt estimation
def LME(y,na, nb, nepoch):
    var_e = 0.0
    cov_theta = 0.0
    N = len(y)
    n = na + nb
    #y = y - np.mean(y)

    l_max = max(na,nb)
    #print(l_max)
    num = np.zeros(l_max + 1)
    den = np.zeros(l_max + 1)
    num[0] = 1
    den[0] = 1

    #print(num)
    #print(den)

    delta = 1e-6
    epsilon = 1e-3
    u = 0.01

    # simulating errors
    sys = (den, num, 1)
    _, e = signal.dlsim(sys, y)
    SSE = np.dot(e.T, e)
    running_SSE = []

    for epoch in range(nepoch):

        # step 1
        X = np.zeros(shape=(N, n))
        for i in range(1, na + 1):
            den_temp = copy.deepcopy(den)
            den_temp[i] = den[i] + delta
            sys = (den_temp, num, 1)
            _, e_theta = signal.dlsim(sys, y)
            X[:, i - 1] = (e - e_theta)[:,0] / delta

        for i in range(1, nb + 1):
            num_temp = copy.deepcopy(num)
            num_temp[i] = num[i] + delta
            sys = (den, num_temp, 1)
            _, e_theta = signal.dlsim(sys, y)
            X[:, i + na - 1] =(e - e_theta)[:,0] / delta

        A = np.dot(X.T, X)
        g = np.dot(X.T, e)

        # step 2
        I = np.identity(n)
        delta_theta = np.dot(np.linalg.inv(A + u * I), g)

        # update coefficients
        den_new = copy.deepcopy(den)
        num_new = copy.deepcopy(num)
        for i in range(1,na+1):
            den_new[i] = den[i] + delta_theta[:na, :][i-1]

        for i in range(1,nb+1):
            num_new[i] = num[i] + delta_theta[na:, :][i-1]

        sys = (den_new, num_new, 1)
        _, e_new = signal.dlsim(sys, y)
        SSE_new = np.dot(e_new.T, e_new)
        running_SSE.append(SSE_new[0,0])

        if SSE_new < SSE:
            if np.linalg.norm(delta_theta,2) < epsilon:
                den = den_new
                num = num_new
                e = e_new
                var_e = SSE_new/(N-n)
                cov_theta = var_e*np.linalg.inv(A)
                break
            else:
                u = u/10
                den = den_new
                num = num_new
                e = e_new
                SSE=SSE_new

        else:
            u = u*10
            #print(u)
            if u > 1e10:
                print("Errors in Program!")
                break

        if epoch == nepoch-1:
            print("Errors in Programs!")



    if na > 0:
        for i in range(1,1+na):
            print("AR process estimated parameter a{:} = {:.4f}".format(i, den[i]))
    if nb > 0:
        for i in range(1,1+nb):
            print("MA process estimated parameter b{:} = {:.4f}".format(i, num[i]))


    return den, num, running_SSE, var_e, cov_theta


# 1-step ahead prediction
def one_step_ARMA(y_train, a, b):
    y_train_pred = np.zeros(len(y_train))
    e = np.zeros(len(y_train))
    na = len(a) - 1
    nb = len(b) - 1
    for i in range(len(y_train)-1):
        sum_ar = 0
        sum_ma = 0
        for j in range(1,na+1):
            if (i - j + 1 > 0) or (j - j +1) == 0:
                sum_ar = sum_ar + y_train[i-j+1]*a[j]


        for k in range(1,nb+1):
            if (i - k + 1 > 0) or (i - k +1) == 0:
                sum_ma = sum_ma + b[k]*(y_train[i-k+1]-y_train_pred[i-k+1])
        y_train_pred[i+1] = -sum_ar + sum_ma


        e = y_train - y_train_pred

    return  y_train_pred, e


# h-step prediction (h>1)
def h_step_ARMA(y_train, y_train_pred, length_test, a, b):
    y_test_pred = np.zeros(length_test)
    na = len(a) - 1
    nb = len(b) - 1

    t = len(y_train) - 1
    for h in range(1,length_test + 1):
        sum_ar = 0
        sum_ma = 0
        for i in range(1,na+1):
            if h - i == 1:
                sum_ar = sum_ar + a[i]*y_train_pred[t]
            elif h - i > 1:
                sum_ar = sum_ar + a[i]*y_test_pred[h-i-2]
            else:
                k = i - h
                sum_ar = sum_ar + a[i]*y_train[t-k]
        for i in range(1,nb+1):
            if h - i > 0:
                sum_ma = sum_ma
            else:
                k = i - h
                sum_ma = sum_ma + b[i]*(y_train[t-k]-y_train_pred[t-k])

        y_test_pred[h-2] = -sum_ar + sum_ma

    return y_test_pred

# strength of trend and seasonality
def ts_strength(S, R, T):
    Ft = max([0, 1 - np.var(R)/np.var(T+R)])
    Fs = max([0, 1 - np.var(R)/np.var(S+R)])
    return Ft, Fs