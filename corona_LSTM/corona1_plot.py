# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 18:21:58 2020

@author: ayanca
"""

import math as m
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

def plot(data1,data2,data3,data4,col,field):
    
    fig, ax = plt.subplots(figsize = (8, 6))
    ax.plot(data1[field])
    ax.legend(['Corona Total Death of the World'])
    ax.set_xlabel('Day(s)', fontname="Palatino Linotype", fontsize=12)
    ax.set_ylabel('Corona Total Death', fontname="Palatino Linotype", fontsize=12)
    ax.set_title('Daily Corona Total Death from 01/01/2020 - 4/22/2020',fontname="Palatino Linotype", fontsize=12)
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.show()
    
    '''
    # Create a time series plot.
    plt.figure(figsize = (8, 6))
    plt.plot(data1[field], label = "Corona Total Death " + field)
    plt.xlabel("Day(s)")
    plt.ylabel("Corona Total Death")
    plt.title("Daily Corona Total Death from 01/01/2020 - 4/22/2020")
    plt.legend()
    plt.grid(True)
    plt.show()
    '''

    df = ((data1).iloc[-1,1:]).sort_values(ascending=False).head(10)
    #print (df)

    fig = plt.figure()
    lines = df.plot.bar(figsize = (8, 6))
    plt.xlabel("Countries")
    plt.ylabel("Corona Total Death Ratio Cases")
    plt.title("Corona Total Death till 4/22/2020 in Descending Order")
    #plt.legend()
    plt.grid(True)
    plt.show()

    fig = plt.figure()
    lines = data1[col].plot.line(figsize = (8, 6))
    plt.xlabel("Countries")
    plt.ylabel("Corona Total Death")
    plt.title("Daily Corona Total Death from 01/01/2020 - 4/22/2020")
    plt.legend()
    plt.grid(True)
    plt.show()

    fig = plt.figure()
    lines = ((data1[col]).iloc[-1,:]).sort_values(ascending=False).plot.bar(figsize = (8, 6))
    plt.xlabel("Countries")
    plt.ylabel("Corona Total Death")
    plt.title("Corona Total Death till 4/22/2020")
    #plt.legend()
    plt.grid(True)
    plt.show()

    # Create a time series plot.
    plt.figure(figsize = (8, 6))
    plt.plot(data2[field], label = "Corona New Death "+ field)
    plt.xlabel("Countries")
    plt.ylabel("Corona New Death")
    plt.title("Daily Corona New Death from 01/01/2020 - 4/22/2020")
    plt.legend()
    plt.grid(True)
    plt.show()

    #lines = data2[headernames].plot.line(figsize = (15, 15))    
    
    # Create a time series plot.
    plt.figure(figsize = (8, 6))
    plt.plot(data3[field], label = "Corona Total Cases "+ field)
    plt.xlabel("Day(s)")
    plt.ylabel("Corona Total Cases")
    plt.title("Daily Corona New Cases from 01/01/2020 - 4/22/2020")
    plt.legend()
    plt.grid(True)
    plt.show()

    #lines = data3[headernames].plot.line(figsize = (15, 15))
    #fontname="Palatino Linotype", fontsize=10
    fig, ax = plt.subplots(figsize = (8, 6))
    ax.plot(data3[field])
    ax.legend(['Corona Total Cases of the World'])
    ax.set_xlabel('Day(s)', fontname="Palatino Linotype", fontsize=12)
    ax.set_ylabel('Corona Total Cases', fontname="Palatino Linotype", fontsize=12)
    ax.set_title('Daily Corona Total Cases from 01/01/2020 - 4/22/2020',fontname="Palatino Linotype", fontsize=12)
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.show()
    
    '''
    fig = plt.figure()
    lines = data3[col].plot.line(figsize = (8, 6))
    plt.xlabel("Countries")
    plt.ylabel("Corona Total Cases")
    plt.title("Daily Corona Total Cases from 01/01/2020 - 4/22/2020")
    plt.legend()
    plt.grid(True)
    plt.show()
    '''

    #print (data1.iloc[-1,:])
    df = ((data3).iloc[-1,1:]).sort_values(ascending=False)
    #print (df)

    fig = plt.figure()
    lines = df.plot.bar(figsize = (8, 6))
    plt.xlabel("Countries")
    plt.ylabel("Corona Total Cases Ratio Cases")
    plt.title("Corona Total Cases till 4/22/2020 in Descending Order")
    #plt.legend()
    plt.grid(True)
    plt.show()
    
    fig, ax = plt.subplots(figsize = (8, 6))
    lines = ((data3[col]).iloc[-1,:]).sort_values(ascending=False).plot.bar()
    #ax.legend(['Corona Total Cases of the World'])
    ax.set_xlabel('Countries', fontname="Palatino Linotype", fontsize=12)
    ax.set_ylabel('Total COVID-19 Cases (per country)', fontname="Palatino Linotype", fontsize=12)
    ax.set_title('Corona Total Cases till 4/22/2020',fontname="Palatino Linotype", fontsize=12)
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.show()

    '''
    fig = plt.figure()
    lines = ((data3[col]).iloc[-1,:]).sort_values(ascending=False).plot.bar(figsize = (8, 6))
    plt.xlabel("Countries")
    plt.ylabel("Total COVID-19 Cases (per country)")
    plt.title("Corona Total Cases till 4/22/2020")
    #plt.legend()
    plt.grid(True)
    plt.show()
    '''

    # Create a time series plot.
    plt.figure(figsize = (8, 6))
    plt.plot(data4[field], label = "Corona New Cases "+ field)
    plt.xlabel("Day(s)")
    plt.ylabel("Corona New Cases")
    plt.title("Daily Corona New Cases from 01/01/2020 - 4/22/2020")
    plt.legend()
    plt.grid(True)
    plt.show()

    #print (data1.iloc[-1,:])
    df = ((((data1).iloc[-1,1:])/((data3).iloc[-1,1:])).sort_values(ascending=False))
    #print (df)
    
    fig, ax = plt.subplots(figsize = (10, 8))
    lines = df.plot.bar()
    #ax.legend(['Corona Total Cases of the World'])
    ax.set_xlabel('Countries', fontname="Palatino Linotype", fontsize=12)
    ax.set_ylabel('COVID-19 Death Ratio [Fatalities / Total Cases] per country', fontname="Palatino Linotype", fontsize=12)
    ax.set_title('Corona Total Death Ratio Cases till 4/22/2020 in Descending Orde',fontname="Palatino Linotype", fontsize=12)
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.show()
    
    '''
    fig = plt.figure()
    lines = df.plot.bar(figsize = (8, 6))
    plt.xlabel("Countries")
    plt.ylabel("COVID-19 Death Ratio [Fatalities / Total Cases] per country")
    plt.title("Corona Total Death Ratio Cases till 4/22/2020 in Descending Order")
    #plt.legend()
    plt.grid(True)
    plt.show()
    '''

    #print (data1.iloc[-1,:])
    df = ((((data1[col]).iloc[-1,:])/((data3[col]).iloc[-1,:])).sort_values(ascending=False))
    #print (df)

    fig = plt.figure()
    lines = df.plot.bar(figsize = (10, 8))
    plt.xlabel("Countries")
    plt.ylabel("COVID-19 Death Ratio [Fatalities / Total Cases] per country")
    plt.title("Corona Total Death Ratio Cases till 4/22/2020")
    #plt.legend()
    plt.grid(True)
    plt.show()

def case_death_cases(data,country):
    #data = data.rename(columns={'World':'Death', 1:'Cases'})  
    #data = data.rename({1:'Cases'}, axis = 1) 
    deathFirst = (len(data[data.Death == 0]))
    caseFirst = (len(data[data.Cases == 0]))
    
    gap = deathFirst - caseFirst
    print ("First Death on ", gap, " days")   
    
    fig = plt.figure()
    lines = data.plot.line(figsize = (8, 6))
    plt.xlabel("Day(s)")
    plt.ylabel("Corona Case vs Death")
    plt.title("Case vs Death from 01/01/2020 - 4/22/2020")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    cases = data.Cases.max()
    death = data.Death.max()
    print ("Cases: ",cases," Death: ",death)
    print ("Cases:Death : ", (cases/death))
    
    labels = 'Cases','Death'
    fig = plt.figure(figsize=(8,6))
    #ax = fig.add_axes([0,0,1,1])    
    months = [cases,death]
    plt.bar(labels,months)
    plt.xlabel("Incident(s)")
    plt.ylabel("Number")
    plt.title("Report in Bar-Graph of  - "+ country)
    #plt.grid(True)
    #plt.legend()
    plt.show()

def date_distribution(df):
    fig = plt.figure()
    lines = df.plot.bar(figsize = (8, 6))
    plt.xlabel("Month(s)")
    plt.ylabel("Day(s)")
    plt.title("Total Day(s) of Data")
    #plt.legend()
    plt.grid(True)
    plt.show()
    
def consolidated_death_case_world(df,country):
    #dec = ((df[(pd.DatetimeIndex(df['date']).month) == 12])['World']).sum()
    jan = (df[(pd.DatetimeIndex(df['date']).month) == 1])[country].max()
    feb = (df[(pd.DatetimeIndex(df['date']).month) == 2])[country].max() 
    mar = (df[(pd.DatetimeIndex(df['date']).month) == 3])[country].max() 
    
    plt.figure(figsize=(8,6))    
    # Data to plot
    labels = 'Jan','Feb','Mar'
    sizes = [jan,feb,mar]
    colors = ['skyblue', 'yellowgreen','orange']
    explode = (0, 0, 0)  # explode 1st slice 
    # Plot
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90) 
    plt.axis('equal')
    plt.show()    
    
    print('Jan:',jan,'Feb:',feb,'Mar:',mar)
    
    labels = 'Jan','Feb','Mar'
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_axes([0,0,1,1])    
    months = [jan,feb,mar]
    plt.bar(labels,months)
    plt.xlabel("Month(s)")
    plt.ylabel("Total Death")
    plt.title("Consolidated Death Growth Bar-Graph of - "+ country)
    plt.grid(False)
    #plt.legend()
    plt.show()
    
def monthly_death_case_world(df,country):
    #dec = ((df[(pd.DatetimeIndex(df['date']).month) == 12])['World']).sum()
    jan = (df[(pd.DatetimeIndex(df['date']).month) == 1])[country].max()
    feb = (df[(pd.DatetimeIndex(df['date']).month) == 2])[country].max() - jan
    mar = (df[(pd.DatetimeIndex(df['date']).month) == 3])[country].max() - feb
    
    plt.figure(figsize=(8,6))    
    # Data to plot
    labels = 'Jan','Feb','Mar'
    sizes = [jan,feb,mar]
    colors = ['skyblue', 'yellowgreen','orange']
    explode = (0, 0, 0)  # explode 1st slice 
    # Plot
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90) 
    plt.axis('equal')
    plt.show()    
    
    print('Jan:',jan,'Feb:',feb,'Mar:',mar)
    
    labels = 'Jan','Feb','Mar'
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_axes([0,0,1,1])    
    months = [jan,feb,mar]
    plt.bar(labels,months)
    plt.xlabel("Month(s)")
    plt.ylabel("Total Death")
    plt.title("Monthly Death Growth Bar-Graph of - "+ country)
    plt.grid(False)
    #plt.legend()
    plt.show()
    
def consolidated_total_case_world(df,country):
    #dec = ((df[(pd.DatetimeIndex(df['date']).month) == 12])['World']).sum()
    jan = (df[(pd.DatetimeIndex(df['date']).month) == 1])[country].max()
    feb = (df[(pd.DatetimeIndex(df['date']).month) == 2])[country].max() 
    mar = (df[(pd.DatetimeIndex(df['date']).month) == 3])[country].max()
    
    plt.figure(figsize=(8,6))    
    # Data to plot
    labels = 'Jan','Feb','Mar'
    sizes = [jan,feb,mar]
    colors = ['skyblue', 'yellowgreen','orange']
    explode = (0, 0, 0)  # explode 1st slice 
    # Plot
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90) 
    plt.axis('equal')
    plt.show()    
    
    print('Jan:',jan,'Feb:',feb,'Mar:',mar)
    
    labels = 'Jan','Feb','Mar'
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_axes([0,0,1,1])    
    months = [jan,feb,mar]
    plt.bar(labels,months)
    plt.xlabel("Month(s)")
    plt.ylabel("Total Case(s)")
    plt.title("Consolidated Case(s) Growth Bar-Graph of  - "+ country)
    plt.grid(False)
    #plt.legend()
    plt.show()
    
def monthly_total_case_world(df,country):
    #dec = ((df[(pd.DatetimeIndex(df['date']).month) == 12])['World']).sum()
    jan = (df[(pd.DatetimeIndex(df['date']).month) == 1])[country].max()
    feb = (df[(pd.DatetimeIndex(df['date']).month) == 2])[country].max() - jan
    mar = (df[(pd.DatetimeIndex(df['date']).month) == 3])[country].max() - feb
    
    plt.figure(figsize=(8,6))    
    # Data to plot
    labels = 'Jan','Feb','Mar'
    sizes = [jan,feb,mar]
    colors = ['skyblue', 'yellowgreen','orange']
    explode = (0, 0, 0)  # explode 1st slice 
    # Plot
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90) 
    plt.axis('equal')
    plt.show()    
    
    print('Jan:',jan,'Feb:',feb,'Mar:',mar)
    
    labels = 'Jan','Feb','Mar'
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_axes([0,0,1,1])    
    months = [jan,feb,mar]
    plt.bar(labels,months)
    plt.xlabel("Month(s)")
    plt.ylabel("Total Case(s)")
    plt.title("Monthly Case(s) Growth Bar-Graph of  - "+ country)
    plt.grid(False)
    #plt.legend()
    plt.show()
    
def correlation_analysis(data):
    #'pearson', 'spearman', 'kendall'
    sns.set_style('whitegrid')
    correlations = data.corr()    
    #print (correlations)
    fig = plt.figure(figsize=(20,16))
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0,9,1)
    ax.set_xticks(ticks)
    ax.set_xticklabels(data.columns)
    ax.set_yticklabels(data.columns)
    sns.heatmap(correlations, annot = True, cmap='coolwarm',linewidths=.1)
    plt.savefig('figure_1.png', dpi=300)
    plt.show() 
    
    
def data_visualization(data):
    #set frame
    fig = plt.figure(figsize = (16, 10))
    ax = fig.gca()

    #histogram
    data.hist(ax = ax)
    plt.show()
    #pyplot.savefig('hist.png')
    #density
    data.plot(kind='density', figsize= (24, 12), subplots=True, layout=(5,5), sharex=False)
    plt.show()
    #boxplot
    data.plot(kind = 'box', figsize=(24, 12), subplots = True, layout = (5,5), sharex = False, sharey = False)
    plt.show()
    #scatter
    scatter_matrix(data, alpha=0.2, figsize=(24, 12), diagonal='kde')
    plt.show()
    
def regression_plot(df):
    fig = plt.figure(figsize=(8,6))    
    sns.regplot(x='Population', y='Cases', data=df)
    plt.show(fig)
    
    fig = plt.figure(figsize=(8,6))
    sns.regplot(x='Population', y='Death', data=df)
    plt.show(fig)
    
    fig = plt.figure(figsize=(8,6))
    sns.regplot(x='Cases', y='Death', data=df)
    plt.show(fig)
    
    fig = plt.figure(figsize=(8,6))
    sns.regplot(x='Sunshine-Mar', y='Death', data=df)
    plt.show(fig)
    
    fig = plt.figure(figsize=(8,6))
    sns.regplot(x='Rainfall-Mar', y='Death', data=df)
    plt.show(fig)
    
    fig = plt.figure(figsize=(8,6))
    sns.regplot(x='Temp-Mar', y='Death', data=df)
    plt.show(fig)
    
def monthly_distribution_cases(df, country):
    jan = (df[(pd.DatetimeIndex(df['date']).month) == 1])[country].astype(float)
    feb = (df[(pd.DatetimeIndex(df['date']).month) == 2])[country].astype(float)
    mar = (df[(pd.DatetimeIndex(df['date']).month) == 3])[country].astype(float)
    
    print ('Cases:')
    print('Jan:',jan.max(),'Feb:',feb.max(),'Mar:',mar.max())
    '''
    kwargs = dict(histtype='stepfilled', alpha=1, normed=True, bins=1)
    plt.hist(jan, **kwargs)
    plt.hist(feb, **kwargs)
    plt.hist(mar, **kwargs)
    plt.xlabel("Cases - Jan, Feb, Mar")
    plt.ylabel("Distribution")
    plt.show()
    print (mar.size)
    '''
    dist_plot_cases(jan,feb,mar,country)
    
    #counts, bin_edges = np.histogram(df, bins=5)
    #print(counts)

def monthly_distribution_death(df, country):
    jan = (df[(pd.DatetimeIndex(df['date']).month) == 1])[country].astype(float)
    feb = (df[(pd.DatetimeIndex(df['date']).month) == 2])[country].astype(float)
    mar = (df[(pd.DatetimeIndex(df['date']).month) == 3])[country].astype(float)
    
    print ('Deathes:')
    print('Jan:',jan.max(),'Feb:',feb.max(),'Mar:',mar.max())
    '''
    kwargs = dict(histtype='stepfilled', alpha=1, normed=True, bins=1)
    plt.hist(jan, **kwargs)
    plt.hist(feb, **kwargs)
    plt.hist(mar, **kwargs)
    plt.xlabel("Death - Jan, Feb, Mar")
    plt.ylabel("Distribution")
    plt.show()
    '''
    print (mar.size)
    dist_plot_death(jan,feb,mar,country)
  
    #counts, bin_edges = np.histogram(df, bins=5)
    #print(counts)    
    
def general_plot(data1,data3,country):
    fig = plt.figure()
    lines = data3[country].plot.bar(figsize = (16, 8))
    plt.xlabel("Day(s)")
    plt.ylabel("Corona Total Cases")
    plt.title("Corona Total Cases till 3/23/2020 at "+country)
    plt.legend()
    plt.grid(True)
    plt.show()
    
    fig = plt.figure()
    lines = data1[country].plot.bar(figsize = (16, 8))
    plt.xlabel("Day(s)")
    plt.ylabel("Corona Total Death")
    plt.title("Corona Total Death till 3/23/2020 at "+country)
    plt.legend()
    plt.grid(True)
    plt.show()
    
def compare_plot(data1,data3):
    
    fig, ax = plt.subplots(figsize = (10, 8))
    ax.plot(data3)
    ax.legend(data3.columns)
    ax.set_xlabel('Day(s)', fontname="Palatino Linotype", fontsize=12)
    ax.set_ylabel('Corona Total Cases', fontname="Palatino Linotype", fontsize=12)
    ax.set_title('Corona Total Growth Cases till 4/22/2020',fontname="Palatino Linotype", fontsize=12)
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.grid(True)
    plt.show()
    
    '''
    fig = plt.figure()
    lines = data3.plot.line(figsize = (8, 6))
    plt.xlabel("Day(s)")
    plt.ylabel("Corona Total Cases")
    plt.title("Corona Total Growth Cases till 4/22/2020")
    plt.legend()
    plt.grid(True)
    plt.show()
    '''
    
    fig = plt.figure()
    lines = data1.plot.line(figsize = (8, 6))
    plt.xlabel("Day(s)")
    plt.ylabel("Corona Total Death")
    plt.title("Corona Total Death Growth till 4/22/2020")
    plt.legend()
    plt.grid(True)
    plt.show()
    
def dist_plot_death(jan,feb,mar,country):    
    sns.set(style="white", palette="muted", color_codes=True)       
    # Set up the matplotlib figure   
    plt.title("Case Distribution Over Month Till Date "+ country)
    f, axes = plt.subplots(2, 2, figsize=(16, 8), sharex=True)
    sns.despine(left=True)
    plt.setp(axes, yticks=[])   
    sns.distplot(jan, hist=False, color="g", kde_kws={"shade": True}, ax=axes[0, 0])
    plt.xlabel("Jan")
    plt.ylabel("Distributions")
    sns.distplot(feb, hist=False, color="g", kde_kws={"shade": True}, ax=axes[0, 1])
    plt.xlabel("Feb")
    plt.ylabel("Distributions")
    sns.distplot(mar, hist=False, color="g", kde_kws={"shade": True}, ax=axes[1, 0])
    plt.xlabel("March - first 2 Weeks")
    plt.ylabel("Distributions")
    sns.distplot(mar, hist=False, color="g", kde_kws={"shade": True}, ax=axes[1, 1])
    plt.xlabel("March - Next 2 Weeks")
    plt.ylabel("Distributions")
    plt.legend()
    plt.tight_layout()
    plt.show()

def dist_plot_cases(jan,feb,mar,country):    
    sns.set(style="white", palette="muted", color_codes=True) 
    # Set up the matplotlib figure    
    plt.title("Death Distribution Over March Till Date "+country)
    f, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True)
    sns.despine(left=True)
    plt.setp(axes, yticks=[])   
    sns.distplot(mar.iloc[0:14], hist=False, color="g", kde_kws={"shade": True}, ax=axes[0])
    plt.xlabel("March - First 2 Weeks")
    plt.ylabel("Distributions")
    sns.distplot(mar.iloc[14:], hist=False, color="g", kde_kws={"shade": True}, ax=axes[1])
    plt.xlabel("March - Next 2 Weeks")
    plt.ylabel("Distributions")
    #sns.distplot(mar.iloc[14:21], hist=False, color="g", kde_kws={"shade": True}, ax=axes[1, 0])
    #sns.distplot(mar.iloc[21:], hist=False, color="g", kde_kws={"shade": True}, ax=axes[1, 1])
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def duration_case(df1,country):
    first = len(df1[df1[country] < 100001])
    second = len(df1[df1[country] < 200001]) - first
    third = len(df1[df1[country]< 300001]) - second - first
    forth = len(df1[df1[country]< 400001]) - third - second - first
    fifth = len(df1[df1[country]< 500001]) - forth - third - second - first
    
    plt.figure(figsize=(8,6))    
    # Data to plot
    labels = 'First','Second','Third','Forth','Fifth'
    sizes = [first,second,third,forth,fifth]
    colors = ['skyblue', 'yellowgreen','orange','red','blue']
    explode = (0, 0, 0, 0, 0)  # explode 1st slice 
    # Plot
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90) 
    plt.axis('equal')
    plt.show()    
    
    print(first,second,third,forth)
    
    labels = '1st 0.1 million','2nd 0.1 million','3rd 0.1 million','4th 0.1 million', '5th 0.1 million'
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_axes([0,0,1,1])    
    days = [first,second,third,forth,fifth]
    plt.barh(labels,days)
    plt.xlabel("Duration [days] for infection of 0.1 million persons")
    plt.ylabel("Total Cases/0.1 Million")
    plt.title("Consolidated Case(s) Growth Bar-Graph of The World ")
    plt.grid(False)
    #plt.legend()
    plt.show()

def duration_death(df1,country):
    #print(df1.World)
    first = len(df1[df1[country] < 5001])
    second = len(df1[df1[country] < 10001]) - first
    third = len(df1[df1[country]< 15001]) - second - first
    forth = len(df1[df1[country]< 20001]) - third - second - first
    
    plt.figure(figsize=(8,6))    
    # Data to plot
    labels = 'First','Second','Third','Forth'
    sizes = [first,second,third,forth]
    colors = ['skyblue', 'yellowgreen','orange','red']
    explode = (0, 0, 0,0)  # explode 1st slice 
    # Plot
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90) 
    plt.axis('equal')
    plt.show()    
    
    print(first,second,third,forth)
    
    labels = 'First','Second','Third','Forth'
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_axes([0,0,1,1])    
    days = [first,second,third,forth]
    plt.barh(labels,days)
    plt.ylabel("Day(s)")
    plt.xlabel("Total Death/ FiveTthousands")
    plt.title("Consolidated Death(s) Growth Bar-Graph of The World")
    plt.grid(False)
    #plt.legend()
    plt.show()