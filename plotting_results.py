# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 20:23:08 2021

@author: leona
"""

import pandas as pd
import numpy as np



def return_series(file_name,n_series):
    results = pd.read_csv('Results\\'+file_name, delimiter = ";", header=None)
    results.columns = ['N','Payoff','Max Payoff','Payoff Std']
    series_payoff = []
    series_max_payoff = []
    series_std_payoff = []
    for i in range(0,n_series):    
        series_payoff.append(results.iloc[i:][results.iloc[i:].reset_index().index % 3 == 0]['Payoff'])
        series_max_payoff.append(results.iloc[i:][results.iloc[i:].reset_index().index % 3 == 0]['Max Payoff'])
        series_std_payoff.append(results.iloc[i:][results.iloc[i:].reset_index().index % 3 == 0]['Payoff Std'])
    return series_payoff,series_max_payoff,series_std_payoff

file_name = 'table_14.txt'
results = pd.read_csv('Results\\'+file_name, delimiter = ";", header=None)
results.columns = ['N','Payoff','Max Payoff','Payoff Std']

payoff,max_payoff,std_payoff = return_series(file_name,3)
models_names = ['Becker','Becker return', 'CNN return']

N_series = results[results.reset_index().index % 3 == 0]['N']


#To append new results:
# file_name = 'table_10.txt'
# results_2 = pd.read_csv('Results\\'+file_name, delimiter = ";", header=None)
# payoff.append(results_2[1])
# models_names.append('CNN_less neurons')
#max_payoff.append(results_2[2])



import matplotlib.pyplot as plt

for line in payoff:
    plt.plot(N_series,line)
plt.xlabel('Number of Bermudan steps (N)')
plt.ylabel('Average payoff')
plt.legend(models_names)
plt.show()

for line in std_payoff:
    plt.plot(N_series,line)
plt.xlabel('Number of Bermudan steps (N)')
plt.ylabel('Std payoff')
plt.legend(models_names)
plt.show()


plt.plot(N_series,(np.asarray(payoff[2])-np.asarray(payoff[0]))/np.asarray(payoff[2])*100)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Number of Bermudan steps (N)')
plt.ylabel('Diff CNN and Becker Payoff')
plt.legend(['CNN percentual increase','Zero line'])
plt.show()

plt.plot(N_series,(np.asarray(std_payoff[2])-np.asarray(std_payoff[0]))/np.asarray(std_payoff[2])*100)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Number of Bermudan steps (N)')
plt.ylabel('Diff CNN and Becker Std dev')
plt.legend(['CNN percentual increase','Zero line'])
plt.show()


