#%% ------------------------------ Import Packages and Libraries -----------------------------------------------------------------------
import numpy as np
import pandas as pd



#%% ------------------------------------ Load dataset ------------------------------------------------------------------

df = pd.read_excel("../data/AirQualityUCI.xlsx")


#%% ------------------------------------ Identifying missing data ---------------------------------------------------------
# define a function to check for missing data
def nan_checker(df):
    """
    Parameters
    ----------
    df : dataframe

    Returns
    ----------
    The dataframe of variables with missing data,
    the proportion of missing data and dtype of variable
    """

    # Get the dataframe of variables with NaN, their proportion of NaN and dtype
    df_nan = pd.DataFrame([[var, df[var].isna().sum() / df.shape[0], df[var].dtype]
                           for var in df.columns if df[var].isna().sum() > 0],
                          columns=['var', 'proportion', 'dtype'])

    # Sort df_nan in accending order of the proportion of NaN
    df_nan = df_nan.sort_values(by='proportion', ascending=False).reset_index(drop=True)

    return df_nan

# since missing values are replaced by -200, we need to convert them back into NaN
df.replace(to_replace= -200, value= np.NaN, inplace= True)

# checking for missing data in dataset
df_nan = nan_checker(df)
print(df_nan)


#%% ------------------------------------- Remove missing data ----------------------------------------------------------
# check for numerical variables that the majority of data is missing (proportion of NaN > 80%)
df_rm = df_nan[(df_nan['dtype']== 'float64') & (df_nan['proportion'] > 0.8)].reset_index(drop=True)
print(df_rm)

# remove numerical variables with a lot of missing data since they do not contribute to the prediction
df.drop(df_rm['var'], axis= 1, inplace= True)

# remaining variables with missing data
df_miss = df_nan[-df_nan['var'].isin(df_rm['var'])].reset_index(drop=True)

#%% ------------------------------------ Impute missing data ------------------------------------------------------------
# we fill missing data using the average of available values in a day
for var in df_miss['var']:
    df[var] = df.groupby("Date")[var].transform(lambda x: x.fillna(x.mean()))

# check if there are still missing values since there may be no data recorded in a day
print(nan_checker(df))

# There are still missing values due to no records in a particular date
# Assume that the current date data is related to the previous and next date data
# We fill NaN of current date using the average of two closest available data

for var in df_miss['var']:
    df[var] = (df[var].fillna(method='ffill', inplace = False) + df[var].fillna(method='bfill', inplace = False))/2

# recheck for missing data
print(nan_checker(df))

#%% ------------------------------------- Handling Datetime Variable -----------------------------------------------------
# we combinate the hour and the date into 1 column

hr = "00:00:00"
for i in range(len(df)):
    # add hour to date
    df.loc[i,"Date"] = str(df.loc[i,"Date"]).replace(hr, str(df.loc[i,"Time"]))
df["Date"] = pd.to_datetime(df["Date"]) # convert to datetime variable
df.drop('Time', axis=1, inplace=True) # drop the column of hour data

#%% --------------------------------- Summarize data and save preprocessed version ----------------------------------------
# print the dimension of df_train
#print(pd.DataFrame([[df.shape[0], df.shape[1]]], columns=['# rows', '# columns']))
df.info() # recheck the data
df.to_csv(r'../data/Preprocessed_AirQuality.csv', index = False) # save the preprocessed data




