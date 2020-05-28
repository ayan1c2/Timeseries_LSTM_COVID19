# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 21:55:44 2020

@author: ayanca
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 18:14:04 2020

@author: ayanca
"""

# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
import time
import matplotlib
from numpy import newaxis
from numpy import concatenate
from pandas import concat
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from LSTM_Models import vanilla_LSTM, stacked_LSTM, bidirectional_LSTM, mul_layer_1, mul_layer_2, mul_layer_3

col_group = ['World']

field = 'World'

path1 = r"C:\Users\ayanca\.spyder-py3\corona_LSTM\ourworldindata.org\total_cases.csv"
data1 = pd.read_csv(path1, na_values="?", low_memory=False)

data1['date_series'] = pd.to_datetime(data1['date'])

data1_1 = data1[col_group]

dataset = data1_1
test_set = data1[field]

training_set = dataset.values
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

#print (training_set_scaled.shape)

x_train = []
y_train = []

train_size = int(len(training_set) * 0.97)
timestamp = len(training_set) - train_size

print (train_size, timestamp)

length = len(training_set)
for i in range(timestamp, length):
    x_train.append(training_set_scaled[i-timestamp:i, 0])
    y_train.append(training_set_scaled[i, 0])
    
x_train = np.array(x_train)
y_train = np.array(y_train)

#print (x_train.shape,y_train.shape)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#model = vanilla_LSTM(x_train.shape[1], 1) #RMSE: 37.639
#model = stacked_LSTM(x_train.shape[1], 1) # RMSE: 36.981
#model = bidirectional_LSTM(x_train.shape[1], 1) # RMSE: 41.731
#model = mul_layer_1(x_train.shape[1], 1)
#model = mul_layer_2(x_train.shape[1], 1)
model = mul_layer_3(x_train.shape[1], 1)

model.fit(x_train, y_train, epochs = 100, batch_size = 10, validation_split=0.05)

#############################################testing####################################

y_test = test_set.iloc[timestamp:].values

cases = (test_set.values).reshape(-1, 1)
cases_scaled = sc.transform(cases)

# the model will predict the values on x_test
x_test = [] 
length = len(test_set)

for i in range(timestamp, length):
    x_test.append(cases_scaled[i-timestamp:i, 0])
    
x_test = np.array(x_test)
print (x_test.shape)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# predicting the stock price values
start = time.time()
y_pred = model.predict(x_test)
#print(y_pred)
print ('compilation time : ', (time.time() - start)*1000.0)
predicted_value = sc.inverse_transform(y_pred)
#print(predicted_value)

# calculate Metric
mae = mean_absolute_error(predicted_value, y_test)
mse = mean_squared_error(predicted_value, y_test)
rmse = sqrt(mean_squared_error(predicted_value, y_test))
forecast_error = np.mean(np.subtract(predicted_value, y_test))
#r2_value = r2_score(predicted_value, y_test,multioutput='variance_weighted')
r2_value = r2_score(predicted_value, y_test)
print('Test MAE: %.3f' % mae)
print('Test MSE: %.3f' % mse)
print('Test RMSE: %.3f' % rmse)
print('Test FE: %.3f' % forecast_error)
print('Test R2: %.3f' % r2_value)

#print (predicted_value)

# plotting the results
fig, ax = plt.subplots()
plt.plot(y_test, color = 'blue', label = 'Actual Total Case(s)')
plt.plot(predicted_value, color = 'red', label = 'Predicted Total Case(s)')
#plt.title('Case Prediction - mul_layer_3 Model')
#plt.xlabel('Time')
#plt.ylabel('Total Case(s)')
ax.set_xlabel('Time', fontname="Palatino Linotype")
ax.set_ylabel('Total Corona Case(s)', fontname="Palatino Linotype")
ax.set_title('Case Prediction - mul_layer_3 Model',fontname="Palatino Linotype")
ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.legend()
plt.show()

###############################################forecast ends#####################################