import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error,mean_absolute_percentage_error


data=pd.read_csv(r"C:\Users\BrunoNad\Downloads\Dataset Fuel Oil(Tabelle1).csv",sep=";")
data=data.iloc[:,[0,1,2,4,5,6,8,10,11,12,14,16,19,20,23,24,25]]
data.columns = data.columns.str.strip()


# start 29.8.2025 22:30
# stop 31.8.2025 6:55

data1 = data[['Timestamp.1','Speed over Ground [knots]','Heading [degrees]','Shaft RPM PS  [rpm]',
               'Shaft RPM SB [rpm]','Shaft Power PS [kW]','Shaft Power SB  [kW]',
               'Shaft Torque PS  [kNm]','Shaft Torque SB [kNm]']]

#print(data1)
data1=data1.dropna()
#print(data1)

# important: type object -> type float 
for col in data1.columns[1:]:
    if data1[col].dtype == 'object':
        data1[col] = pd.to_numeric(data1[col].str.replace(',', '.'), errors='coerce')
data1['Timestamp.1'] = pd.to_datetime(data1['Timestamp.1'], format='%d.%m.%Y %H:%M')
print(data1)
data1=data1.drop([974])
data1.reset_index(inplace=True)
data1=data1.drop('index',axis=1) 
print(data1)

######################################################################################################################################################
###########################################################################
###########################################################################
###########################################################################
# zadnja znamenka 0 
# 4 i 6 -> 5 (mean)
# 8 i 0 i 2 -> 0 (mean)

""" def round_time_to_5min(dt):
    discard = pd.Timedelta(minutes=dt.minute % 5, seconds=dt.second, microseconds=dt.microsecond)
    dt -= discard
    if discard >= pd.Timedelta(minutes=2.5):
        dt += pd.Timedelta(minutes=5)
    return dt

def create_custom_dataframe_rounded_time(df):
    new_cols = {}
    length = len(df)
    for col in df.columns[1:]:
        new_col_values = []
        i = 0
        toggle_inner = True

        while i < length:
            if i == 0:
                new_col_values.append(df.iloc[i][col])
                i += 1
            else:
                if toggle_inner:
                    end = min(i+2, length)
                    mean_val = df.iloc[i:end][col].mean()
                    new_col_values.append(mean_val)
                    i += 2
                else:
                    end = min(i+3, length)
                    mean_val = df.iloc[i:end][col].mean()
                    new_col_values.append(mean_val)
                    i += 3
                toggle_inner = not toggle_inner

        new_cols[col] = new_col_values
    new_times = []
    i = 0
    toggle = True
    while i < length:
        if i == 0:
            orig_time = df.iloc[i]['Timestamp.1']
            new_times.append(round_time_to_5min(orig_time))
            i += 1
        else:
            if toggle:
                times_group = df.iloc[i:i+2]['Timestamp.1']
                avg_time = times_group.mean()
                new_times.append(round_time_to_5min(avg_time))
                i += 2
            else:
                times_group = df.iloc[i:i+3]['Timestamp.1']
                avg_time = times_group.mean()
                new_times.append(round_time_to_5min(avg_time))
                i += 3
            toggle = not toggle

    new_df = pd.DataFrame(new_cols)
    new_df.insert(0, 'Timestamp.1', new_times)

    return new_df

data1_new=create_custom_dataframe_rounded_time(data1)
#print(data1_new)
 """
######################################################################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################

# zadnja znamenka minuta = 0
# zadnja znamenka minuta in [2,8] -> ne uzima se taj red u obzir
# zadnja znamenka minuta in [4,6] -> računa se aritm.sredina i zadnja znamenka minuta je 5
""" 
def process_dataframe(df):
    df = df.copy()
    df['Timestamp'] = pd.to_datetime(df.iloc[:, 0])
    df = df.set_index('Timestamp')

    new_rows = []
    new_index = []

    i = 0
    while i < len(df):
        minute_last_digit = df.index[i].minute % 10

        if minute_last_digit == 0:
            new_rows.append(df.iloc[i].values[1:])
            new_index.append(df.index[i])
            i += 1

        elif minute_last_digit in [2, 8]:
            i += 1

        elif minute_last_digit in [4, 6]:
            if i + 1 < len(df):
                avg_values = (df.iloc[i].values[1:] + df.iloc[i + 1].values[1:]) / 2
                time1 = df.index[i]
                time2 = df.index[i + 1]
                mid_time = time1 + (time2 - time1) / 2
                minute = mid_time.minute
                rounded_minute = 5 * round(minute / 5)
                mid_time = mid_time.replace(minute=rounded_minute, second=0, microsecond=0)
                new_rows.append(avg_values)
                new_index.append(mid_time)
                i += 2
            else:
                i += 1
    new_df = pd.DataFrame(new_rows, columns=df.columns[1:], index=new_index)
    new_df.reset_index(inplace=True)
    new_df.rename(columns={'index': df.columns[0]}, inplace=True)
    return new_df

data1_new=process_dataframe(data1)

 """

#print(data1_new)
######################################################################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################

# zadnja znamenka minuta = 0
# zadnja znamenka minuta in [2,4,6,8] -> računa se aritm.sredina i zadnja znamenka minuta je 5

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
print(data1_new)


######################################################################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################






# concatenation: data1_new, data_wind, data_fuel

data_wind=data.loc[:-1,'Wind Speed [m/s]'].dropna()
data_wind=pd.to_numeric(data_wind.str.replace(',', '.'))
#print(data_wind)

data_fuel=data.loc[:-1,'Fuel Consumpt. (TOTAL) [l/h]'].dropna()
data_fuel=pd.to_numeric(data_fuel.str.replace(',', '.'))
data_fuel=data_fuel[12:].reset_index().drop('index',axis=1)
#print(data_fuel)

table=pd.concat([data1_new,data_wind,data_fuel],axis=1)
print(table) 

#print(table.describe()) 
















""" 
fig=plt.figure(figsize=[20,8])
fig.subplots_adjust(hspace=0.4,wspace=0.4)
for i in range(2,len(table.columns)+1):
    ax=fig.add_subplot(4,4,i)
    sns.histplot(data=table,x=table[table.columns[i-1]],kde=True,color="g")
plt.savefig("C:/Users/BrunoNad/Documents/Project_consumption/distributions.png")
plt.show()


fig = plt.figure(figsize=[25,10])
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(2,len(table.columns)+1):
    ax = fig.add_subplot(4,4,i)
    sns.boxplot(x=table[table.columns[i-1]],color="red")
plt.savefig("C:/Users/BrunoNad/Documents/Project_consumption/boxplots.png")
plt.show()

plt.figure(figsize=[24,14])
sns.heatmap(table.corr(),annot=True,fmt='.2f',cmap='coolwarm')
plt.title("Correlation")
plt.savefig("C:/Users/BrunoNad/Documents/Project_consumption/matrix_correlation.png")
plt.show()
 """

""" X=table.drop('Fuel Consumpt. (TOTAL) [l/h]',axis=1)
y=table['Fuel Consumpt. (TOTAL) [l/h]']
print(X)

train_end = int(X.shape[0] * 0.7)
val_end = int(X.shape[0] * 0.85)
Xtrain=X[:train_end].drop('Timestamp.1',axis=1)
Xvalid=X[train_end:val_end].drop('Timestamp.1',axis=1)
Xtest=X[val_end:].drop('Timestamp.1',axis=1)
ytrain=y[:train_end]
yvalid=y[train_end:val_end]
ytest=y[val_end:] """



""" def add_lags_target(df,target_column,lag_steps):
    for i in range(1,lag_steps+1):
        df[f'{target_column}_lag_{i}'] = df[target_column].shift(i)
    return df

table_lags3=add_lags_target(table,'Fuel Consumpt. (TOTAL) [l/h]',3).dropna()

print(table_lags3) """


# LINEAR REGRESSION 

""" lr=LinearRegression()
lr.fit(Xtrain,ytrain)
predlr=lr.predict(Xtest)
print(pd.DataFrame({'predicted':predlr,'actual':ytest,'diff':abs(predlr-ytest),'percent error':(abs(predlr-ytest)/ytest)*100}))

print("Mean error (in %): ",sum((abs(predlr-ytest)/ytest)*100)/len(predlr))
print("MAE: ",mean_absolute_error(ytest,predlr))
print(mean_absolute_percentage_error(ytest,predlr)*100) """


# DECISION TREE REGRESSOR
""" 
dt=RandomForestRegressor(n_estimators=100)
dt.fit(Xtrain,ytrain)
print(mean_absolute_error(dt.predict(Xtest),ytest))
 """

""" data_lat_long=data.iloc[:,[1,2]].dropna().reset_index().drop('index',axis=1)
print(data_lat_long)

for col in data_lat_long.columns[1:]:
    if data_lat_long[col].dtype == 'object':
        data_lat_long[col] = pd.to_numeric(data_lat_long[col].str.replace(',', '.'), errors='coerce')


plt.figure(figsize=(16, 8))
plt.scatter(data_lat_long.loc[:15,'longitude'], data_lat_long.loc[:15,'latitude'], color='blue')
plt.plot(data_lat_long.loc[:15,'longitude'], data_lat_long.loc[:15,'latitude'], color='red',linewidth=0.6)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.savefig("C:/Users/BrunoNad/Documents/Project_consumption/ship.png")
plt.show() """


""" 
def rate_of_turn(ih,fh,time):
    diff=(fh-ih+180)%360-180
    return diff/time

headings_df=data1_new['Heading [degrees]']

rot= [rate_of_turn(headings_df[i], headings_df[i+1],5) for i in range(15)]
rot_df = pd.DataFrame({'rate_of_turn': rot})
print(rot_df)

 """