import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error,mean_absolute_percentage_error,make_scorer
from sklearn.model_selection import GridSearchCV,TimeSeriesSplit
import xgboost as xgb
#import tensorflow
#import keras_tuner
#from tensorflow import keras
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense,BatchNormalization,Dropout,Input,LSTM,GRU
#from tensorflow.keras.callbacks import EarlyStopping
#from tensorflow.keras.regularizers import l2
#from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from scipy.stats import pearsonr


data=pd.read_csv(r"C:\Users\BrunoNad\Downloads\Dataset Fuel Oil(Tabelle1).csv",sep=";")

data=data.iloc[:,[0,1,2,4,5,6,8,10,11,12,14,16,19,20,23,24,25]]
data.columns = data.columns.str.strip()
data1 = data[['Timestamp.1','Speed over Ground [knots]','Heading [degrees]','Shaft RPM PS  [rpm]',
               'Shaft RPM SB [rpm]','Shaft Power PS [kW]','Shaft Power SB  [kW]',
               'Shaft Torque PS  [kNm]','Shaft Torque SB [kNm]']]
data1=data1.dropna()
#print(data1)

# type object -> type float 
for col in data1.columns[1:]:
    if data1[col].dtype == 'object':
        data1[col] = pd.to_numeric(data1[col].str.replace(',', '.'), errors='coerce')


data1['Timestamp.1'] = pd.to_datetime(data1['Timestamp.1'], format='%d.%m.%Y %H:%M')

def dataframe_new(df):
    df = df.copy()
    df['Timestamp'] = pd.to_datetime(df.iloc[:, 0])
    df = df.set_index('Timestamp')
    new_rows = []
    new_index = []
    zero_minute_indices = [i for i, t in enumerate(df.index) if t.minute % 10 == 0]

    for idx in range(len(zero_minute_indices) - 1):
        start_idx = zero_minute_indices[idx]
        end_idx = zero_minute_indices[idx + 1]
        new_rows.append(df.iloc[start_idx].values[1:])
        new_index.append(df.index[start_idx])
        if end_idx > start_idx + 1:
            intermediate_values = df.iloc[start_idx + 1:end_idx].values[:, 1:]
            avg_values = intermediate_values.mean(axis=0)
        else:
            avg_values = np.full(df.shape[1] - 1, np.nan)
        time1 = df.index[start_idx]
        time2 = df.index[end_idx]
        mid_time = time1 + (time2 - time1) / 2
        rounded_minute = 5 * round(mid_time.minute / 5)
        mid_time = mid_time.replace(minute=rounded_minute, second=0, microsecond=0)
        new_rows.append(avg_values)
        new_index.append(mid_time)
    last_zero_idx = zero_minute_indices[-1]
    new_rows.append(df.iloc[last_zero_idx].values[1:])
    new_index.append(df.index[last_zero_idx])
    new_df = pd.DataFrame(new_rows, columns=df.columns[1:], index=new_index)
    new_df.reset_index(inplace=True)
    new_df.rename(columns={'index': df.columns[0]}, inplace=True)

    return new_df


data1_new = dataframe_new(data1)

data_wind=data['Wind Speed [m/s]'].dropna()
data_wind=data_wind[:-1]
data_wind=pd.to_numeric(data_wind.str.replace(',', '.'))


data_fuel=data['Fuel Consumpt. (TOTAL) [l/h]'].dropna()
data_fuel=pd.to_numeric(data_fuel.str.replace(',', '.'))
data_fuel=data_fuel[12:-1].reset_index().drop('index',axis=1)


table=pd.concat([data1_new,data_wind,data_fuel],axis=1)

print(table)



def add_lags_target(df,target_column,lag_steps):
    for i in range(1,lag_steps+1):
        df[f'{target_column}_lag_{i}'] = df[target_column].shift(i)
    return df


table=add_lags_target(table,'Fuel Consumpt. (TOTAL) [l/h]',12).dropna().reset_index(drop=True)


X=table.drop(['Timestamp.1','Fuel Consumpt. (TOTAL) [l/h]'],axis=1)
y=table['Fuel Consumpt. (TOTAL) [l/h]']

def pearson_correlation(df, target_column):
    correlations = {}
    for col in df.columns:
        if col != target_column:
            corr, p_value = pearsonr(df[col], df[target_column])
            correlations[col] = {'pearson_corr': corr, 'p_value': p_value}
    return pd.DataFrame(correlations).T

result=pearson_correlation(table.iloc[:,1:],'Fuel Consumpt. (TOTAL) [l/h]')

# Train-test-split 
train_end=int(X.shape[0]*0.7)
Xtrain=X[:train_end]
ytrain=y[:train_end]
Xtest=X[train_end:]
ytest=y[train_end:]


################################################  
# LINEAR REGRESSION
lr=LinearRegression()
lr.fit(Xtrain,ytrain)
predlr=lr.predict(Xtest)

#print("MAE: ",round(mean_absolute_error(predlr,ytest),2),"\t MSE: ",round(mean_squared_error(predlr,ytest),2),"\t MAPE: ",round(mean_absolute_percentage_error(predlr,ytest)*100,2),"%")
# MAE:  53.22 	 MSE:  6712.75 	 MAPE:  8.16 %

# pd.DataFrame({'column':Xtrain.columns,'coefficient':lr.coef_})

# lr.intercept_  
# 287.57644561993806
################################################



################################################ 
# SVR

svr=SVR(kernel='linear')
svr.fit(Xtrain,ytrain)
#pd.DataFrame({'actual':ytest,'predicted':svr.predict(Xtest),'diff':abs(ytest-svr.predict(Xtest))}).to_csv("C:/Users/BrunoNad/Documents/Project_consumption/results_svr.csv")
#print("MAE: ",round(mean_absolute_error(ytest,svr.predict(Xtest)),2),"\t MSE: ",round(mean_squared_error(svr.predict(Xtest),ytest),2),"\t MAPE: ",round(mean_absolute_percentage_error(svr.predict(Xtest),ytest)*100,2),"%")
#  MAE:  48.19      MSE:  5494.97   MAPE:  7.2 %





################################################