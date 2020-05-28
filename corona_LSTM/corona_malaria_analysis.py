# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 20:32:46 2020

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

col_group_1 = ['total_cases','total_deaths','Ratio']
col_group = ['location','Ratio']

path1 = r"C:\Users\ayanca\.spyder-py3\corona_LSTM\ourworldindata.org\full_data1.csv"
data1 = pd.read_csv(path1, na_values="?", low_memory=False)
print (data1)
data = data1
df1 = data[col_group]
print (df1)
#data1['Ratio'].sort_values(ascending=True).plot.barh(figsize = (25, 25))

df1 = df1.sort_values(by=['Ratio'], ascending=True)
#df1 = df1[col_group]
print (df1)