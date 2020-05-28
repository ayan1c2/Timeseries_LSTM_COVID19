# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 16:20:46 2020

@author: ayanca
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

days = 100
population = 200000
spread_factor = 5.0
days_to_recover = 10
initally_afflicted = 5

town = pd.DataFrame(data={'id': np.arange(population), 'infected': False, 'recovery_day': None, 'recovered': False})
town = town.set_index('id')

#print (city)

firstCases = town.sample(initally_afflicted, replace=False)
town.loc[firstCases.index, 'infected'] = True
town.loc[firstCases.index, 'recovery_day'] = days_to_recover

#print (city)

stat_active_cases = [initally_afflicted]
stat_recovered = [0]

for day in range(1, days):
    # Indicate people who have recovered on 'day'
    town.loc[town['recovery_day'] == day, 'recovered'] = True
    town.loc[town['recovery_day'] == day, 'infected'] = False
    
    # Calcuate the number of people who are infected on 'day'
    spreadingPeople = town[ (town['infected'] == True)]
    totalCasesToday = round(len(spreadingPeople) * spread_factor)
    casesToday = town.sample(totalCasesToday, replace=True)
    # Avoid people who were already infected in cases on 'day'
    casesToday = casesToday[ (casesToday['infected'] == False) & (casesToday['recovered'] == False) ]
    # Indicate the new cases as infected, and their recovery day
    town.loc[casesToday.index, 'infected'] = True
    town.loc[casesToday.index, 'recovery_day'] = day + days_to_recover

    stat_active_cases.append(len(town[town['infected'] == True]))
    stat_recovered.append(len(town[town['recovered'] == True]))
    
    
'''   
print (days)
plt.figure(figsize = (8, 6))
plt.bar(x=np.arange(days), height=stat_active_cases, color="green")
plt.xlabel("Day(s)")
plt.ylabel("Corona Active Cases")
plt.title("spread_factor - 5.0")
plt.legend()
plt.grid(True)
plt.show()   
'''


fig, ax = plt.subplots(figsize = (8, 6))
plt.bar(x=np.arange(days), height=stat_active_cases, color="green")
ax.set_xlabel('Day(s)', fontname="Palatino Linotype")
ax.set_ylabel('Corona Active Cases', fontname="Palatino Linotype")
ax.set_title('spread_factor - 5.0',fontname="Palatino Linotype")
ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.grid(True)
plt.legend()
plt.show()
