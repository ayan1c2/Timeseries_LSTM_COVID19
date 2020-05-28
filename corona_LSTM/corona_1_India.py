# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:49:13 2020

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
from corona1_plot import plot, date_distribution, consolidated_death_case_world, consolidated_total_case_world, monthly_death_case_world, monthly_total_case_world, case_death_cases, correlation_analysis, data_visualization, regression_plot, monthly_distribution_cases, monthly_distribution_death, general_plot, compare_plot

col_1 = ['State_Union_Territories','ConfirmedIndianNational','ConfirmedForeignNational','Cured','Deaths']
col = ['State_Union_Territories', 'NumPrimaryHealthCenters_HMIS', 'NumCommunityHealthCenters_HMIS', 'NumSubDistrictHospitals_HMIS', 'NumDistrictHospitals_HMIS', 'TotalPublicHealthFacilities_HMIS', 'NumPublicBeds_HMIS', 'NumRuralHospitals_NHP18', 'NumRuralBeds_NHP18', 'NumUrbanHospitals_NHP18', 'NumUrbanBeds_NHP18']
col_2 = ['State_Union_Territories','Beds']
col_3 = ['State_Union_Territories']
extra_field = 'All India'
field = 'India'
state = ''


# Original data set retrieved from here:
path1 = r"C:\Users\ayanca\.spyder-py3\corona_LSTM\covid19-in-india\covid_19_india.csv"
data1 = pd.read_csv(path1, na_values="?", low_memory=False, index_col = 0)
#data1 = pd.read_csv(path1, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

path2 = r"C:\Users\ayanca\.spyder-py3\corona_LSTM\covid19-in-india\HospitalBedsIndia.csv"
data2 = pd.read_csv(path2, na_values="?", low_memory=False, index_col = 0)

path3 = r"C:\Users\ayanca\.spyder-py3\corona_LSTM\covid19-in-india\population_india_census2011.csv"
data3 = pd.read_csv(path3, na_values="?", low_memory=False, index_col = 0)

print (data1.head())

#data1 = data1[col_1]
data1_1 = data1.iloc[:,1:].groupby(['State_Union_Territories']).max().astype(float)
print (data1_1.head())

fig = plt.figure()
data1_1.sort_values(by=['Deaths'], ascending=False).plot.bar(figsize = (24, 15))
plt.ylabel("Count(s)")
plt.xlabel("State_Union_Territories")
plt.title("Corona Effect in India till Date")
plt.legend()
plt.grid(False)
plt.show()
plt.close

'''
data2_2 = data2[:-1]
data2_2 = data2_2[col]


fig = plt.figure()
data2_2.plot.bar(figsize = (20, 10))
plt.ylabel("Hospital Bed(s)")
plt.xlabel("State_Union_Territories")
plt.title("Corona Effect in India till 3/22/2020")
plt.legend()
plt.grid(False)
plt.show()

data2_2['Beds'] = data2_2.iloc[:,1:].sum(axis=1)
data2_2 = data2_2[col_2]
print(data2_2.head())


fig2 = plt.figure()
data2_2.plot.bar(figsize = (20, 10))
plt.ylabel("Total Hospital Bed(s)")
plt.xlabel("State_Union_Territories")
plt.title("Corona Effect in India")
plt.legend()
plt.grid(False)
plt.show()
'''