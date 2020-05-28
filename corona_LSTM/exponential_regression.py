# -*- coding: utf-8 -*-
"""
Created on Sat May 16 17:00:03 2020

@author: ayanca
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib

'''
x_data = np.array([10, 20, 30, 40, 50])
y_data = np.array([1, 3, 5, 7, 9])

log_x_data = np.log(x_data)
log_y_data = np.log(y_data)

curve_fit = np.polyfit(log_x_data, y_data, 1)
print(curve_fit)

y = 4.84 * log_x_data - 10.79
plt.plot(log_x_data, y_data, "o")
plt.plot(log_x_data, y)
'''

'''
Factors_3.xls
Case-Jan	Case-Feb	Case-Mar	Death-Jan	Death-Feb	Death-Mar
9771, 83964, 304916, 213, 2910, 14153
'''

x_data = np.array([9771, 83964, 304916])
y_data = np.array([213, 2910, 14153])

log_x_data = np.log(x_data)
log_y_data = np.log(y_data)

curve_fit = np.polyfit(x_data, log_y_data, 1)
print(curve_fit)

y = np.exp(5.95734475e+00) * np.exp(1.25996126e-05*x_data)

'''
plt.figure(figsize = (8, 6))
plt.xlabel("Case(s)")
plt.ylabel("Death(s)")
plt.plot(x_data, y_data, "*")
plt.plot(x_data, y, label='y ≈ e^(5.95734475e+00) * e^(1.25996126e-059*x)')
plt.legend()
plt.grid(True)
plt.show()
'''

fig, ax = plt.subplots(figsize = (10, 8))
ax.plot(x_data, y_data, "*")
plt.plot(x_data, y, label='y ≈ e^(5.95734475e+00) * e^(1.25996126e-059*x)')
plt.legend(['Current Count','Future Prediction [y ≈ e^(5.95734475e+00) * e^(1.25996126e-059*x)]'], fontsize=12)
ax.set_xlabel('Case(s)', fontname="Palatino Linotype", fontsize=12)
ax.set_ylabel('Death(s)', fontname="Palatino Linotype", fontsize=12)
#ax.set_title('Corona Total Death Ratio Cases till 4/22/2020 in Descending Orde',fontname="Palatino Linotype", fontsize=12)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.grid(True)
plt.show()

'''
from sklearn import linear_model

rng = np.random.RandomState(1)
x = 10000 * rng.rand(50)
y = x - 500 + 500*rng.randn(50)
df = pd.DataFrame({'x':x,'y':y})
g = sns.lmplot('x','y',df,fit_reg=True,aspect=1.5,ci=None,scatter_kws={"s": 100})

regr = linear_model.LinearRegression()

X = df.x.values.reshape(-1,1)
y = df.y.values.reshape(-1,1)

regr.fit(X, y)
print(regr.coef_[0])
print(regr.intercept_)

g = sns.lmplot('x','y',df,fit_reg=True,aspect=1.5,ci=None, scatter_kws={"s": 100})
props = dict(boxstyle='round', alpha=0.5,color=sns.color_palette()[0])
textstr = 'y=-499.3 + 1.0x'
g.ax.text(0.0, 0.0, textstr, transform=g.ax.transAxes, fontsize=14, bbox=props)
'''
