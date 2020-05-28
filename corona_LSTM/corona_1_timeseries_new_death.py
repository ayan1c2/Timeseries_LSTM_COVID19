# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 17:12:46 2020

@author: ayanca
"""

import warnings
import math as m
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import array
from pandas import DataFrame
from math import sqrt
from numpy import concatenate
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from LSTM_Models import vanilla_LSTM, stacked_LSTM, bidirectional_LSTM, mul_layer_1

field = 'World'
col_group = [field]

path1 = r"C:\Users\ayanca\.spyder-py3\corona_LSTM\ourworldindata.org\total_deaths - New.csv"
data1 = pd.read_csv(path1, na_values="?", low_memory=False)

data1['date_series'] = pd.to_datetime(data1['date'])
data1 = data1[col_group]

# choose a number of time steps
n_steps = 3
df = []
df2 = (data1[field]).iloc[-n_steps:]
print(df2)
input = array(df2.values)
print(input)
#print (data1.head())

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
raw_seq = data1[field]
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
# define model
model = vanilla_LSTM(n_steps, n_features)
#model = stacked_LSTM(n_steps, n_features) # RMSE: 36.981
#model = bidirectional_LSTM(n_steps, n_features) # RMSE: 41.731
#model = mul_layer_1(n_steps, n_features)
# fit model
model.fit(X, y, epochs=100, verbose=0)
# demonstrate prediction
x_input = input
#x_input = array([12941, 14601, 17498])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)

for i in range(7):
    input = input.reshape((1, n_steps, n_features))
    yhat = model.predict(input, verbose=0)
    #print(yhat)
    df.append(yhat)
    #print(df)
    input = np.append(input,yhat)
    #print(input)
    input = input[-n_steps:]
    #print (input)

df =  pd.DataFrame.from_records(df) 
print (df)

result = (pd.concat([data1[field], df], axis=0, sort=False)).astype(int)
print(result.max())

#data2[field].plot.line()
#result.plot.line()

result.plot.bar(color = 'red', label = 'Predicted Death(s)')
data1[field].plot.line(color = 'blue', label = 'Actual Death(s)')
plt.title('Death(s) Prediction for '+field)
plt.xlabel('Time')
plt.ylabel('Death')
plt.legend()
plt.show()