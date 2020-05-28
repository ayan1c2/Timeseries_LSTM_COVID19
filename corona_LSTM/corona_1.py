# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 21:17:23 2020

@author: ayanca
"""

import warnings
import math as m
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from LSTM_Models import vanilla_LSTM, stacked_LSTM, bidirectional_LSTM, mul_layer_1
from corona1_plot import plot, date_distribution, consolidated_death_case_world, consolidated_total_case_world, monthly_death_case_world, monthly_total_case_world, case_death_cases, correlation_analysis, data_visualization, regression_plot, monthly_distribution_cases, monthly_distribution_death, general_plot, compare_plot, duration_death, duration_case
from statsmodels.tsa.stattools import adfuller

#headernames = ['China', 'Italy', 'Spain','Germany','Iran','France','South Korea','Switzerland','Netherlands','Norway','India']
col = ['Netherlands','China', 'Italy','Spain','Germany','Canada','France','Switzerland','Belgium','Brazil','India','United States','United Kingdom','Turkey','Russia','Iran','Portugal']
col_group = ['date','World']
col_group_research = ['India','Singapore','Turkey','Iran']


field = 'World'
country = 'World'
recorved = 12

# load dataset
def parser(x):
	return pd.to_datetime(x)

# Original data set retrieved from here:
path1 = r"C:\Users\ayanca\.spyder-py3\corona_LSTM\ourworldindata.org\total_deaths.csv"
data1 = pd.read_csv(path1, na_values="?", low_memory=False)
#data1 = pd.read_csv(path1, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

path2 = r"C:\Users\ayanca\.spyder-py3\corona_LSTM\ourworldindata.org\new_deaths.csv"
data2 = pd.read_csv(path2, na_values="?", low_memory=False)

path3 = r"C:\Users\ayanca\.spyder-py3\corona_LSTM\ourworldindata.org\total_cases.csv"
data3 = pd.read_csv(path3, na_values="?", low_memory=False)

path4 = r"C:\Users\ayanca\.spyder-py3\corona_LSTM\ourworldindata.org\new_cases.csv"
data4 = pd.read_csv(path4, na_values="?", low_memory=False)

#path5 = r"C:\Users\ayanca\.spyder-py3\corona_LSTM\Factors_2.csv"
#data5 = pd.read_csv(path5, na_values="?", low_memory=False)

path6 = r"C:\Users\ayanca\.spyder-py3\corona_LSTM\Factors_3.csv"
data6 = pd.read_csv(path6, na_values="?", low_memory=False)

warnings.filterwarnings('ignore')
np.seterr(divide = 'ignore') 
np.seterr(all = 'ignore') 

#plt.plot(data)
#plt.show()

print ('Death India: ',data1[field].max())
print ('Cases India: ',data3[field].max())
print ('Death Percent: ', ((data1[field].max())/(data3[field].max()))*100)

duration_death(data1[['World']],field)
duration_case(data3[['World']],field)
'''
result = adfuller(data1[['World']])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:', np.mean(data1[['World']]), np.std(data1[['World']]))
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
    
result = adfuller(data2[['World']])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:', np.mean(data2[['World']]), np.std(data2[['World']]))
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
    
result = adfuller(data3[['World']])
print('ADF Statistic: %f' % result[0])
print('Critical Values:', np.mean(data3[['World']]), np.std(data3[['World']]))
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
    
result = adfuller(data4[['World']])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:', np.mean(data4[['World']]), np.std(data4[['World']]))
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

'''
print ('Death: ',data1[field].max())
print ('Cases: ',data3[field].max())

#print("Countries: ", data5['Country'])
encoder = LabelEncoder()
#data5['Country'] = encoder.fit_transform(data5['Country'])
#data6['Country'] = encoder.fit_transform(data6['Country'])
#data_visualization(data5)
data_visualization(data6)
#regression_plot(data5)
regression_plot(data6)

#general_plot(data1, data3, field)
compare_plot(data1[col_group_research], data3[col_group_research])

#print (data5.head(), data5.columns)
print (data6.head(), data6.columns)

#correlation_analysis(data5)
correlation_analysis(data6)

#print("Data Information: ", data5.info())
#print("Describe data: ", data5.describe()) #statistical summary of the data


plot(data1,data2,data3,data4,col,field)

data = data1[col_group]
data['date'] = (pd.to_datetime(data['date']))
#print ((pd.DatetimeIndex(data1['date']).month).unique())

date_distribution(((pd.DatetimeIndex(data['date']).month)).value_counts())
consolidated_death_case_world(data,country)
monthly_death_case_world(data,country)

data = data3[col_group]
data['date'] = (pd.to_datetime(data['date']))
#print ((pd.DatetimeIndex(data1['date']).month).unique())
consolidated_total_case_world(data,country)
monthly_total_case_world(data,country)

data1['Death']= data1[field]
data3['Cases']= data3[field]
case_death_cases(pd.concat([data1.Death,data3.Cases], axis=1, sort=False),field)

#convert_to_monthly_Data(data1,data3)
 

monthly_distribution_cases(data3,field)
monthly_distribution_death(data1,field)