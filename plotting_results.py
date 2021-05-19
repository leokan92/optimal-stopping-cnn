# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 20:23:08 2021

@author: leona
"""

import pandas as pd
import numpy as np
import seaborn as sns



def return_series(file_name,n_series):
    results = pd.read_csv('Results\\SP\\'+file_name, delimiter = ";", header=None)
    results.columns = ['N','Payoff','Max Payoff','Payoff Std']
    series_payoff = []
    series_max_payoff = []
    series_std_payoff = []
    for i in range(0,n_series):    
        series_payoff.append(results.iloc[i:][results.iloc[i:].reset_index().index % n_series == 0]['Payoff'])
        series_max_payoff.append(results.iloc[i:][results.iloc[i:].reset_index().index % n_series == 0]['Max Payoff'])
        series_std_payoff.append(results.iloc[i:][results.iloc[i:].reset_index().index % n_series == 0]['Payoff Std'])
    return series_payoff,series_max_payoff,series_std_payoff

file_name = 'SP500_2000.txt'
number_of_models = 2
results = pd.read_csv('Results\\SP\\'+file_name, delimiter = ";", header=None)
results.columns = ['N','Payoff','Max Payoff','Payoff Std']

payoff,max_payoff,std_payoff = return_series(file_name,number_of_models)
#models_names = ['Becker','Becker return', 'CNN return']
models_names = ['Becker', 'CNN return']
N_series = results[results.reset_index().index % number_of_models == 0]['N']


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

mean = (np.asarray(payoff[number_of_models-1])-np.asarray(payoff[0]))/np.asarray(payoff[number_of_models-1])*100

plt.plot(N_series,mean)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Number of Bermudan steps (N)')
plt.ylabel('Diff CNN and Becker Payoff')
plt.legend(['CNN percentual increase','Zero line'])
plt.show()


std = 100*(np.asarray(std_payoff[number_of_models-1])+np.asarray(std_payoff[0]))/np.asarray(payoff[number_of_models-1])+ 100*np.asarray(std_payoff[number_of_models-1])/(np.asarray(payoff[0])+np.asarray(payoff[number_of_models-1]))

       
plt.plot(N_series,(std))
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Number of Bermudan steps (N)')
plt.ylabel('Diff CNN and Becker Std dev')
plt.legend(['CNN percentual increase','Zero line'])
plt.show()


plt.errorbar(range(0,len(mean)), mean, std, fmt='-o')
plt.show()


plt.plot(N_series,(np.asarray(payoff[2])/np.asarray(payoff[2])))
plt.plot(N_series,(np.asarray(payoff[2])/np.asarray(payoff[2]))+np.asarray(std_payoff[2])/np.asarray(payoff[2]))
plt.plot(N_series,(np.asarray(payoff[2])/np.asarray(payoff[2]))-np.asarray(std_payoff[2])/np.asarray(payoff[2]))
plt.plot(N_series,(np.asarray(payoff[0])/np.asarray(payoff[2]))+np.asarray(std_payoff[0])/np.asarray(payoff[2]))
plt.plot(N_series,(np.asarray(payoff[0])/np.asarray(payoff[2]))-np.asarray(std_payoff[0])/np.asarray(payoff[2]))
plt.plot(N_series,(np.asarray(payoff[0])/np.asarray(payoff[2])))
plt.xlabel('Number of Bermudan steps (N)')
plt.ylabel('Diff CNN and Becker Std dev')
plt.legend(['CNN percentual increase'])
plt.show()

def convert_array_tensor(arr):
    list_arr = []
    for i in arr:
        list_arr.append(i.item())
    return np.asarray(list_arr)

beckerh = np.load('Results/SP/BeckeH_10.npy',allow_pickle=True)
beckerh = convert_array_tensor(beckerh)
cnn = np.load('Results/SP/Becker_cnn_10.npy',allow_pickle=True)
cnn = convert_array_tensor(cnn)
time = range(0,len(cnn))
dataset = pd.DataFrame({'index': time,'BecherH': beckerh, 'Becker_cnn': cnn}, columns=['BecherH', 'Becker_cnn'])

sns.displot(dataset, x="BecherH", kind="kde", fill=True)



