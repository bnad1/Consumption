import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error,mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV
#import xgboost as xgb
#import tensorflow
#import keras_tuner
#from tensorflow import keras
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense,BatchNormalization,Dropout,Input,LSTM,GRU
#from tensorflow.keras.callbacks import EarlyStopping
#from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from sklearn.svm import SVR,LinearSVR

data=pd.read_csv(r"C:\Users\BrunoNad\Downloads\Dataset Fuel Oil(Tabelle1).csv",sep=";")
data=data.iloc[:,[0,1,2,4,5,6,8,10,11,12,14,16,19,20,23,24,25]]
data.columns = data.columns.str.strip()

data1 = data[['Timestamp.1','Speed over Ground [knots]','Heading [degrees]','Shaft RPM PS  [rpm]',
               'Shaft RPM SB [rpm]','Shaft Power PS [kW]','Shaft Power SB  [kW]',
               'Shaft Torque PS  [kNm]','Shaft Torque SB [kNm]']]
data1=data1.dropna()

data1=data1.iloc[::5].reset_index(drop=True)

for col in data1.columns[1:]:
    if data1[col].dtype == 'object':
        data1[col] = pd.to_numeric(data1[col].str.replace(',', '.'), errors='coerce')

data1.loc[:,'Timestamp.1']=pd.to_datetime(data1['Timestamp.1'], format='%d.%m.%Y %H:%M')

data_wind=data['Wind Speed [m/s]'].dropna()
data_wind=data_wind.iloc[::2].reset_index(drop=True)
data_wind=pd.to_numeric(data_wind.str.replace(',', '.'))

data_fuel=data['Fuel Consumpt. (TOTAL) [l/h]'].dropna()
data_fuel=pd.to_numeric(data_fuel.str.replace(',', '.'))
data_fuel=data_fuel[12:].reset_index().drop('index',axis=1)
data_fuel=data_fuel.iloc[::2].reset_index(drop=True)

table=pd.concat([data1,data_wind,data_fuel],axis=1)

def pearson_correlation(df, target_column):
    correlations = {}
    for col in df.columns:
        if col != target_column:
            corr, p_value = pearsonr(df[col], df[target_column])
            correlations[col] = {'pearson_corr': corr, 'p_value': p_value}
    return pd.DataFrame(correlations).T

result=pearson_correlation(table.iloc[:,1:],'Fuel Consumpt. (TOTAL) [l/h]')


def add_lags_target(df,target_column,lag_steps):
    for i in range(1,lag_steps+1):
        df[f'{target_column}_lag_{i}'] = df[target_column].shift(i)
    return df

table=add_lags_target(table,'Fuel Consumpt. (TOTAL) [l/h]',5).dropna().reset_index(drop=True)

X=table.drop(['Timestamp.1','Fuel Consumpt. (TOTAL) [l/h]'],axis=1)
y=table['Fuel Consumpt. (TOTAL) [l/h]']

train_end=int(X.shape[0]*0.7)
Xtrain=X[:train_end]
ytrain=y[:train_end]
Xtest=X[train_end:]
ytest=y[train_end:]

################################################ 
# Linear regression
lr=LinearRegression()
lr.fit(Xtrain,ytrain)
predlr=lr.predict(Xtest)
#print("MAE: ",round(mean_absolute_error(predlr,ytest),2),"\t MSE: ",round(mean_squared_error(predlr,ytest),2),"\t MAPE: ",round(mean_absolute_percentage_error(predlr,ytest)*100,2),"%")
# MAE:  44.8 	 MSE:  3156.08 	 MAPE:  7.65 %

# lr.intercept_
# 291.48583616784373
################################################ 



###################################
# SVR
""" param_grid = {'C': [0.1, 1, 10, 100],'epsilon':[0.01,0.1,0.2,0.5],'kernel':['linear']}
svr=SVR()
grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(Xtrain, ytrain)
print("best parameters: ",grid_search.best_estimator_)
print("best score: ",-grid_search.best_score_) """


svr=SVR(kernel='linear')
svr.fit(Xtrain,ytrain)
#pd.DataFrame({'actual':ytest,'predicted':svr.predict(Xtest),'error':abs(ytest-svr.predict(Xtest))}).to_csv("C:/Users/BrunoNad/Documents/Project_consumption/results_svr10min.csv")
#pd.DataFrame({'actual':ytest,'predicted':svr.predict(Xtest),'error':abs(ytest-svr.predict(Xtest))}).to_excel("C:/Users/BrunoNad/Documents/Project_consumption/results_svr10min.xlsx")
print("SVR model\n MAE: ",round(mean_absolute_error(ytest,svr.predict(Xtest)),2),"\t MSE: ",round(mean_squared_error(svr.predict(Xtest),ytest),2),"\t MAPE: ",round(mean_absolute_percentage_error(svr.predict(Xtest),ytest)*100,2),"%")

import pickle
#pkl_filename="C:/Users/BrunoNad/Documents/Project_consumption/svr_model.pkl"

pkl_filename="svr_model.pkl"
# save model
with open(pkl_filename,'wb') as file:
  pickle.dump(svr,file)

# load model
with open(pkl_filename,'rb') as file:
  Svr=pickle.load(file) 

# MAE:  39.39      MSE:  2754.87   MAPE:  6.67 %
##################################



