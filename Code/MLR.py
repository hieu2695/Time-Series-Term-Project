import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.holtwinters as ets
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from scipy import stats
from MyFunctions import Q_val_cal, ACF_error, ACF_plot, autocorrelation_cal, series_autocorrelation_cal, corr_cal
import warnings
warnings.filterwarnings("ignore")


#%%----------
data = pd.read_csv("Preprocessed_AirQuality.csv", index_col="Date", parse_dates=True)
target = "NO2(GT)"  # target variable
features = np.setdiff1d(data.columns, [target]).tolist()   # features




#%% Splitting

# split the data into training and testing sets
df_train, df_test = train_test_split(data, shuffle=False, test_size=0.2)

#%% ----- Scaling
# The StandardScaler
ss = StandardScaler()
# fit and standardize the training data
train = pd.DataFrame(ss.fit_transform(df_train), columns=df_train.columns)
# apply standardization to testing data
test = pd.DataFrame(ss.transform(df_test), columns=df_test.columns)

#%% Getting feature matrix and target variable
X_train = train[features] # getting the feature matrix for the training set
Y_train = train[target] # the target variable for the training set

X_test = test[features]  # getting the feature matrix for the testing set
Y_test = test[target] # the target variable for the testing set


#%% ------------ Correlation

# correlation matrix
corr_matrix = data.corr()

# plt correlation matrix
ax = sns.heatmap(corr_matrix,vmin =-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20,220,n=200),
            square = True, annot = False)
bottom, top =ax.get_ylim()
ax.set_ylim(bottom+0.5,top-0.5)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, horizontalalignment='right')
plt.show()

#%% -------------- Mannual LME using normal equation
# number of samples
T = X_train.shape[0]
# number of features
k = len(features)
# we will create X matrix with T rows and k+1 columns
X = []

# the 1st column of matrix X are 1s
for i in range(T):
    X.append(1)
X = [X]

# other rows
for var in features:
    X.append(X_train[var])

# transform X into a matrix
X = np.array(X).T

# create matrix Y
Y = np.array([Y_train]).T

# calculate the coefficient using LSE equation
B = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),Y)
print("Matrix of regression coefficients is:\n", B,"\n")
count = 1
print("The intercept is: ", B[0,0])
for var in X_train.columns:
    print("The coefficient of "+var+" is: ", B[count,0])
    count = count +1

#%% using OLS
# since the model need an intercept, we add a column of 1s to X_train
X_train = sm.add_constant(X_train)

# fit the model to training set
model = sm.OLS(Y_train, X_train).fit()

print("\n",model.summary(),"\n")
print("Adj R2 : ", model.rsquared_adj)
print("AIC : ", model.aic)
print("BIC : ", model.bic)

# forecast testing set
X_test = sm.add_constant(X_test)
forecasts = model.predict(X_test)

# transform forecasts to original scale
test_copy = test.copy(deep=True)
test_copy[target] = forecasts
test_copy = pd.DataFrame(ss.inverse_transform(test_copy),
                              columns=test_copy.columns)
forecasts = test_copy[target]

#%%
# plot the prediction
inds = data.index
train_inds, test_inds = train_test_split(inds, test_size=0.2, shuffle=False)
plt.figure()
plt.plot(train_inds[-100:], df_train[target][-100:], label ="Training set")
plt.plot(test_inds[:200],df_test[target][:200], label="Testing set")
plt.plot(test_inds[:200],forecasts[:200], label="LSE forecasts")
plt.xlabel("Date")
plt.xticks(rotation=30)
plt.ylabel(target)
plt.title("Forecasting of NO2 concentration using LSE")
plt.legend(loc='best')
plt.show()

#%% --------------- Forecast Errors
errors =  np.array(df_test[target]) -np.array(forecasts)
SSE_test = np.square(errors).sum()
MSE_test = np.square(errors).mean()

print("MSE of forecasted values using LME is: ", MSE_test)


#%% -------------- Residuals
predictions = model.predict(X_train)

# transform forecasts to original scale
train_copy = train.copy(deep=True)
train_copy[target] = predictions
train_copy = pd.DataFrame(ss.inverse_transform(train_copy),
                              columns=train_copy.columns)
predictions = train_copy[target]

# residuals
residuals =  np.array(df_train[target]) -np.array(predictions)

SSE_train = np.square(residuals).sum()
MSE_train = np.square(residuals).mean()
print("MSE of fitted values using LME is: ", MSE_train)
print("Mean of residuals is: ", np.mean(residuals))


#%% Stats

est_var_train = np.sqrt(SSE_train/(T-k-1))
print(" The estimated variance of prediction errors is:", est_var_train )
est_var_test = np.sqrt(SSE_test/(len(Y_test)-k-1))
print(" The estimated variance of forecast errors is:", est_var_test )


#%%
title = "ACF plot for residuals"
h = 48
ACF_plot(residuals,h, title)
Q_val = Q_val_cal(residuals, h,T)
print("Q-value of residuals is: ", Q_val)

#%% residual plot
r = corr_cal(np.array(predictions),np.array(df_train[target]))
plt.figure()
sns.regplot(np.array(predictions), np.array(df_train[target]), label="Predictions vs Targets",line_kws={"color": "red"})
plt.title("True values vs Predictions with correlation r={:.4f}".format(r))
plt.xlabel("True values")
plt.ylabel("Fitted values")
plt.show()


r = corr_cal(np.array(predictions),residuals)
plt.figure()
sns.regplot(np.array(predictions), residuals, label="Predictions vs Residuals",line_kws={"color": "red"})
plt.title("Residuals vs Fitted values with correlation r={:.4f}".format(r))
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.show()

#%%--- Backward stepwise using AIC, BIC, Adj R2 ----------------
# create a deep copy of the training set so updates wont affect the original data.
Xtrain = X_train.copy(deep=True)
AIC = model.aic
BIC = model.bic
R2 = model.rsquared_adj
print("\nBackward stepwise using AIC, BIC and Adjusted R_square:")
# threshold for adj R2
# if adj R2 increase while AIC and BIC improved, we definitely update the model
# if adj R2 decreases an amount below this threshold while the AIC and BIC improved,
# we update the model since the decrease of adj R2 does not hurt the model performance
threshold = 0.005

for var in features:
    print("\n" + "-"*10 +" Dropping " + var + "-"*10)
    print("Previous AIC ={:.4f}".format(AIC), ",BIC ={:.4f}".format(BIC), ",Adj_R2 = {:.4f}".format(R2))
    X_opt = Xtrain.drop(var, axis =1, inplace = False)
    model = sm.OLS(Y_train, X_opt).fit()
    print("New AIC ={:.4f}".format(model.aic), ",BIC ={:.4f}".format(model.bic), ",Adj_R2 = {:.4f}".format(model.rsquared_adj))
    R2_diff = np.subtract(R2, model.rsquared_adj)
    if (model.aic < AIC) and (model.bic < BIC) and (R2_diff < threshold):
        print("The feature "+ var +" is not important, we can drop it from the model.")
        Xtrain.drop(var, axis =1, inplace = True)
        AIC = model.aic
        BIC = model.bic
        R2 = model.rsquared_adj
    else:
        print("The feature "+ var +" is important, we need to keep it in the model.")
        model = sm.OLS(Y_train, Xtrain).fit()

features_to_be_eliminated = []
for var in features:
    if var not in Xtrain.columns:
        features_to_be_eliminated.append(var)
print("\nFeatures to be eliminated are ", features_to_be_eliminated)
print("\n",model.summary(),"\n")
print("Adj R2 : ", model.rsquared_adj)
print("AIC : ", model.aic)
print("BIC : ", model.bic)

#%% ----------- Backward stepwise using p-values of t-test ---------
print("\nBackward stepwise using t-test p_values:\n")
Xtrain = X_train.copy(deep=True) # copy the original training set
alpha = 0.05 # significant level aplha - confident level = 95%

# retrain the model with all features
model = sm.OLS(Y_train, X_train).fit()

# get the p values
l = model.pvalues
features_to_be_eliminated = []

while max(l[1:]) > alpha or max(l[1:])  == alpha: # while there is a p-value larger than significane level
    features_copy = l.index # get the feature index
    index = np.argmax(l[1:]) # get the index of the feature with max p-value
    var = features_copy[index+1] # get the name of the feature with max p-value
    print("The t-test p_value for "+var+ " is {:.4f}".format(max(l)))
    print("The regression coefficient of feature "+ var+" is not statistically different than 0.")
    print("Dropping " + var + " from the model ..............")
    features_to_be_eliminated.append(var)
    print(" \n--------------------- Training new regression model ---------------------")

    # drop the feature with the max p-value if p-value > 0.05
    Xtrain.drop(var, axis =1, inplace=True)
    # training new model
    model = sm.OLS(Y_train, Xtrain).fit()
    l = model.pvalues # get the new list of p-values
print("\nBackward stepwise completed!")
print("\nFeatures to be eliminated are ", features_to_be_eliminated)
print("\n",model.summary(),"\n")
print("Adj R2 : ", model.rsquared_adj)
print("AIC : ", model.aic)
print("BIC : ", model.bic)

#%%-------------- Forward stepwise using AIC, BIC, Adj R2 ------
Xtrain = X_train[X_train.columns[0]].copy(deep=True)
# retrain the new model
model1 = sm.OLS(Y_train, Xtrain).fit()
AIC = model1.aic
BIC = model1.bic
R2 = model1.rsquared_adj
print("\nForward stepwise using AIC, BIC and Adjusted R_square:")
threshold = 0.005
features = [var for var in features if var != X_train.columns[0]]
feature_opt = []
feature_opt.append(X_train.columns[0])
feature_not_add = []
for var in features:
    print("\n" + "-"*10 +" Adding " + var + "-"*10)
    feature_opt.append(var) # add var to feature
    Xtrain = X_train[feature_opt].copy(deep=True) # create feature matrix to be trained
    model1 = sm.OLS(Y_train, Xtrain).fit()
    print("New AIC ={:.4f}".format(model1.aic), ",BIC ={:.4f}".format(model1.bic), ",Adj_R2 = {:.4f}".format(model1.rsquared_adj))
    #R2_diff = np.subtract(R2, model.rsquared_adj)
    if (model1.aic < AIC) and (model1.bic < BIC) and (model1.rsquared_adj > R2):
        print("The feature "+ var +" is important, we add it to the model.")
        AIC = model1.aic
        BIC = model1.bic
        R2 = model1.rsquared_adj
    else:
        feature_opt = [x for x in feature_opt if x!= var]
        feature_not_add.append(var)
        print("The feature "+ var +" is not important, we do not add it to the model.")

print("\nFeatures are not added are ", feature_not_add)
print("\n",model1.summary(),"\n")
print("Adj R2 : ", model1.rsquared_adj)
print("AIC : ", model1.aic)
print("BIC : ", model1.bic)

#%% ----------- Forward stepwise using p-values of t-test ---------
print("\nForward stepwise using t-test p_values:\n")
alpha = 0.05 # significant level aplha - confident level = 95%
Xtrain = X_train[X_train.columns[0]].copy(deep=True)


feature_opt = []
feature_opt.append(X_train.columns[0])
feature_not_add = []

for var in features:
    print("\n" + "-"*10 +" Adding " + var + "-"*10)
    feature_opt.append(var) # add var to feature
    Xtrain = X_train[feature_opt].copy(deep=True) # create feature matrix to be trained
    model1 = sm.OLS(Y_train, Xtrain).fit()
    pval= model1.pvalues.loc[var]
    if  pval < alpha :
        print("The feature "+ var +" is important, we add it to the model.")
        AIC = model1.aic
        BIC = model1.bic
        R2 = model1.rsquared_adj
    else:
        feature_opt = [x for x in feature_opt if x!= var]
        feature_not_add.append(var)
        print("The feature "+ var +" causes a feature not significant due to p-value {:.4f}, we do not add it to the model.".format(pval))

print("\nForward stepwise completed!")
print("\nFeatures not added are ", feature_not_add)
print("\n",model1.summary(),"\n")
print("Adj R2 : ", model1.rsquared_adj)
print("AIC : ", model1.aic)
print("BIC : ", model1.bic)

#%% ---------------- Evaluation ------------------
# forecast testing set
features = [x for x in X_test.columns if x not in features_to_be_eliminated]
X_test = X_test[features]
X_test = sm.add_constant(X_test)
forecasts = model.predict(X_test)

# transform forecasts to original scale
test_copy = test.copy(deep=True)
test_copy[target] = forecasts
test_copy = pd.DataFrame(ss.inverse_transform(test_copy),
                              columns=test_copy.columns)
forecasts = test_copy[target]

#%%
# plot the prediction
inds = data.index
train_inds, test_inds = train_test_split(inds, test_size=0.2, shuffle=False)
plt.figure()
plt.plot(train_inds[-100:], df_train[target][-100:], label ="Training set")
plt.plot(test_inds[:200],df_test[target][:200], label="Testing set")
plt.plot(test_inds[:200],forecasts[:200], label="LSE forecasts")
plt.xlabel("Date")
plt.xticks(rotation=30)
plt.ylabel(target)
plt.title("Forecasting of NO2 concentration after feature selection")
plt.legend(loc='best')
plt.show()

#%% --------------- Forecast Errors
errors =  np.array(df_test[target]) -np.array(forecasts)
SSE_test = np.square(errors).sum()
MSE_test = np.square(errors).mean()

print("MSE of forecasted values using LME is: ", MSE_test)


#%% -------------- Residuals
predictions = model.predict(X_train[features])

# transform forecasts to original scale
train_copy = train.copy(deep=True)
train_copy[target] = predictions
train_copy = pd.DataFrame(ss.inverse_transform(train_copy),
                              columns=train_copy.columns)
predictions = train_copy[target]

# residuals
residuals =  np.array(df_train[target]) -np.array(predictions)

SSE_train = np.square(residuals).sum()
MSE_train = np.square(residuals).mean()
print("MSE of fitted values using LME is: ", MSE_train)
print("Mean of residuals is: ", np.mean(residuals))


#%% Stats

est_var_train = np.sqrt(SSE_train/(T-k-1))
print(" The estimated variance of prediction errors is:", est_var_train )
est_var_test = np.sqrt(SSE_test/(len(Y_test)-k-1))
print(" The estimated variance of forecast errors is:", est_var_test )


#%%
title = "ACF plot for residuals after feature selection"
h = 48
ACF_plot(residuals,h, title)
Q_val = Q_val_cal(residuals, h,T)
print("Q-value of residuals is: ", Q_val)

#%% residual plot
r = corr_cal(np.array(predictions),np.array(df_train[target]))
plt.figure()
sns.regplot(np.array(predictions), np.array(df_train[target]), label="Predictions vs Targets",line_kws={"color": "red"})
plt.title("True values vs Predictions with correlation r={:.4f}".format(r))
plt.xlabel("True values")
plt.ylabel("Fitted values")
plt.show()


r = corr_cal(np.array(predictions),residuals)
plt.figure()
sns.regplot(np.array(predictions), residuals, label="Predictions vs Residuals",line_kws={"color": "red"})
plt.title("Residuals vs Fitted values with correlation r={:.4f}".format(r))
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.show()
