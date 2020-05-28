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

field = 'India'
col_group = [field]

#path1 = r"C:\Users\ayanca\.spyder-py3\corona_LSTM\ourworldindata.org\total_deaths.csv"
#data1 = pd.read_csv(path1, na_values="?", low_memory=False)

path2 = r"C:\Users\ayanca\.spyder-py3\corona_LSTM\ourworldindata.org\total_cases - India.csv"
data2 = pd.read_csv(path2, na_values="?", low_memory=False)

#data1['date_series'] = pd.to_datetime(data1['date'])
#data1 = data1[col_group]

#print (data1.head())
# choose a number of time steps
n_steps = 3
data2['date_series'] = pd.to_datetime(data2['date'])
data2 = data2[col_group]
data = data2.values

df = []
df2 = (data2[field]).iloc[-n_steps:]
print(df2)
input = array(df2.values)
print(input)
#print (data2.head())

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
raw_seq = data2[field]
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
# define model
#model = vanilla_LSTM(n_steps, n_features) #513052
#model = stacked_LSTM(n_steps, n_features) # RMSE: 36.981
model = bidirectional_LSTM(n_steps, n_features) # RMSE: 41.731

# fit model
model.fit(X, y, epochs=100, verbose=0)
# demonstrate prediction
x_input = input
#x_input = array([12941, 14601, 17498])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)

for i in range(21):
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

result = (pd.concat([data2[field], df], axis=0, sort=False)).astype(float)
print(result.max())

result.plot.bar(color = 'red', label = 'Predicted Case(s)')
data2[field].plot.line(color = 'blue', label = 'Actual Case(s)')
plt.title('Case(s) Prediction for '+ field)
plt.xlabel('Time')
plt.ylabel('Cases')
plt.legend()
plt.show()