# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 13:06:54 2020

@author: ayanca
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
import time
from numpy import newaxis
from numpy import concatenate
from pandas import concat
from pandas import DataFrame

path1 = r"C:\Users\ayanca\.spyder-py3\corona_LSTM\ourworldindata.org\total_cases.csv"
#path1 = r"C:\Users\ayanca\.spyder-py3\corona_LSTM\ourworldindata.org\total_deaths.csv"  
data1 = pd.read_csv(path1, na_values="?", low_memory=False)

data1['date'] = pd.to_datetime(data1['date'])
#print (data1)

data1 = data1.fillna(0)
#print (data1)
data1.to_csv(path1)

data1 = pd.read_csv(path1, na_values="?", low_memory=False)
data1 = data1.replace(to_replace=0.0, method='ffill')
print (data1)
data1.to_csv(path1)

#data1['India'].plot.line()