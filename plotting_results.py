# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 20:23:08 2021

@author: leona
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


##########################################################################
# Functions for plotting
##########################################################################


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w
def convert_array_tensor(arr):
    list_arr = []
    for i in arr:
        list_arr.append(i.item())
    return np.asarray(list_arr)

def return_series(file_name,n_series,path):
    results = pd.read_csv(path+file_name, delimiter = ";", header=None)
    results.columns = ['N','Payoff','Max Payoff','Payoff Std']
    series_payoff = []
    series_max_payoff = []
    series_std_payoff = []
    for i in range(0,n_series):    
        series_payoff.append(results.iloc[i:][results.iloc[i:].reset_index().index % n_series == 0]['Payoff'])
        series_max_payoff.append(results.iloc[i:][results.iloc[i:].reset_index().index % n_series == 0]['Max Payoff'])
        series_std_payoff.append(results.iloc[i:][results.iloc[i:].reset_index().index % n_series == 0]['Payoff Std'])
    return series_payoff,series_max_payoff,series_std_payoff

def create_dataframe_two_models(path,model_names,N_series):
    df_results = pd.DataFrame(columns= ['Models','N','Payoff'])
    for model in model_names:
        for N in N_series:
            payoffs = np.load(path+model+'_'+N+'.npy',allow_pickle=True)
            payoffs = convert_array_tensor(payoffs)
            df_temp = pd.DataFrame({'Models':[model]*len(payoffs),'N':[N]*len(payoffs),'Average Payoff':payoffs})
            df_results = df_results.append(df_temp)
            df_results['Models'].replace('BeckeH','Becker', inplace=True)
            df_results['Models'].replace('Becker_cnn','CNN', inplace=True)
    return df_results



#############################################################################
# Plotting the Avg Pay with standar deviation for each N on Brownian motion:
############################################################################# 


path = 'Results/BROWNIAN/'
file_name = 'brownian_results.txt'
model_names = 'BeckeH', 'Becker_cnn'

results = pd.read_csv(path+file_name, delimiter = ";", header=None)
results.columns = ['N','Payoff','Max Payoff','Payoff Std']

N_series = results.N.values.astype(str)

df_for_seaborn = create_dataframe_two_models(path,model_names,N_series)


f = plt.figure(figsize=(15,5))
sns.lineplot(data=df_for_seaborn, x="N", y="Average Payoff", hue="Models",style='Models',palette = 'binary',ci='sd')
# pallet options: Set1
f.savefig('brownian_results.pdf', bbox_inches='tight')

#################################################################################
# Plotting the Avg Pay with standar deviation for each N on Harmonic time-series:
################################################################################# 


path = 'Results/HARM/'
file_name = 'harmonic_results.txt'
model_names = 'BeckeH', 'Becker_cnn'

results = pd.read_csv(path+file_name, delimiter = ";", header=None)
results.columns = ['N','Payoff','Max Payoff','Payoff Std']

N_series = results.N.values.astype(str)

df_for_seaborn = create_dataframe_two_models(path,model_names,N_series)

temp = df_for_seaborn.loc[df_for_seaborn['Models']=='Becker']
temp.loc[temp['N']=='10'].std()
temp.loc[temp['N']=='10'].mean()

f = plt.figure(figsize=(15,5))
sns.lineplot(data=df_for_seaborn, x="N", y="Average Payoff", hue="Models",style='Models',palette = 'binary',ci='sd')
#sns.lineplot(data=df_for_seaborn, x="N", y="Payoff", hue="Models",palette = 'binary')
# pallet options: Set1
f.savefig('harmonic_results.pdf', bbox_inches='tight')


########################################################################################################
# Plotting the Avg Pay with standar deviation for each N on fractional Brownian motion time-series:
######################################################################################################## 


path = 'Results/HARM/'
file_name = 'harmonic_results.txt'
model_names = 'BeckeH', 'Becker_cnn'

results = pd.read_csv(path+file_name, delimiter = ";", header=None)
results.columns = ['N','Payoff','Max Payoff','Payoff Std']

N_series = results.N.values.astype(str)

df_for_seaborn = create_dataframe_two_models(path,model_names,N_series)

temp = df_for_seaborn.loc[df_for_seaborn['Models']=='Becker']
temp.loc[temp['N']=='10'].std()
temp.loc[temp['N']=='10'].mean()

f = plt.figure(figsize=(15,5))
sns.lineplot(data=df_for_seaborn, x="N", y="Average Payoff", hue="Models",style='Models',palette = 'binary',ci='sd')
#sns.lineplot(data=df_for_seaborn, x="N", y="Payoff", hue="Models",palette = 'binary')
# pallet options: Set1
f.savefig('harmonic_table.pdf', bbox_inches='tight')


#################################################################################
# Plotting the Avg Pay with standar deviation for each N on Harmonic time-series:
################################################################################# 


path = 'Results/FBM/'
file_name = 'fbm_results.txt'
model_names = 'BeckeH', 'Becker_cnn'

results = pd.read_csv(path+file_name, delimiter = ";", header=None)
results.columns = ['N','Payoff','Max Payoff','Payoff Std']

N_series = results.N.values.astype(str)

df_for_seaborn = create_dataframe_two_models(path,model_names,N_series)

temp = df_for_seaborn.loc[df_for_seaborn['Models']=='Becker']
temp.loc[temp['N']=='10'].std()
temp.loc[temp['N']=='10'].mean()

f = plt.figure(figsize=(15,5))
sns.lineplot(data=df_for_seaborn, x="N", y="Average Payoff", hue="Models",style='Models',palette = 'binary',ci='sd')
#sns.lineplot(data=df_for_seaborn, x="N", y="Payoff", hue="Models",palette = 'binary')
# pallet options: Set1
f.savefig('fbm_table.pdf', bbox_inches='tight')


##########################################################################
# Plotting the Avg Pay for each N on Brownian motion:
##########################################################################    

path = 'Results/BROWNIAN/'

number_of_models = 2
MA_steps = 1


payoff,max_payoff,std_payoff = return_series(file_name,number_of_models,path)
#models_names = ['Becker','Becker return', 'CNN return']
models_names = ['Becker', 'CNN']
N_series = results[results.reset_index().index % number_of_models == 0]['N']



f = plt.figure(figsize=(15,5))
for line in payoff:
    if MA_steps<2:
        plt.plot(N_series,line)
    else:
        plt.plot(N_series[:-(MA_steps-1)],moving_average(line.values, MA_steps))  
plt.title('Evolution of the average payoff by the number of Bermudan Steps|MA('+str(MA_steps)+')')
plt.xlabel('Number of Bermudan steps (N)')
plt.ylabel('Average payoff')
plt.legend(models_names)
plt.show()
f.savefig('brownian_results.pdf', bbox_inches='tight')



#################################################################################
# Plotting distributions of the Monte Carlo simulations
#################################################################################



N = 160

beckerh = np.load('Results/HARM/BeckeH_'+str(N)+'.npy',allow_pickle=True)
beckerh = convert_array_tensor(beckerh)
cnn = np.load('Results/HARM/Becker_cnn_'+str(N)+'.npy',allow_pickle=True)
cnn = convert_array_tensor(cnn)
time = range(0,len(cnn))
dataset = pd.DataFrame({'index': time,'BecherH': beckerh, 'Becker_cnn': cnn}, columns=['BecherH', 'Becker_cnn'])

sns.displot({'Becker':beckerh,'CNN':cnn}, kind="kde", fill=True,legend=True).set_axis_labels('Average Payoff','Density')
#sns.displot(cnn, kind="kde", fill=True,label = 'CNN')
plt.title('Payoff of each batch | Harmonic Time-series | N = '+str(N))
plt.show()

# bins = np.linspace(30, 100, 100)

# plt.hist(beckerh, bins, alpha=0.5, label='Beckers approach')
# plt.hist(cnn, bins, alpha=0.5, label='CNN approach')
# plt.legend(loc='upper right')
# plt.show()


#################################################################################
# Crude oil price plotting
#################################################################################

train_series = np.load('Results/Energy/Becker_cnn_train_30.npy',allow_pickle=True)
val_series = np.load('Results/Energy/Becker_cnn_val_30.npy',allow_pickle=True)


MA_steps = 100

plt.plot(moving_average(train_series,MA_steps),label = 'Training')
plt.plot(moving_average(val_series,MA_steps),label = 'Validation')
plt.legend()
plt.title('Oil prices Training and Validation average payoff / MA('+str(MA_steps)+')')
plt.xlabel('Average Payoff')
plt.ylabel('epochs')
plt.show()

#################################################################################
# Crude oil price plotting - Comparing distributions
#################################################################################
N =  200

cnn = np.load('Results/Energy/Becker_cnn_'+str(N)+'.npy',allow_pickle=True)
cnn = convert_array_tensor(cnn)

max_dist = np.load('Results/Energy/Max_dist_'+str(N)+'.npy',allow_pickle=True)
max_dist = convert_array_tensor(max_dist)

lsmc = np.load('Results/Energy/LSMC_'+str(N)+'.npy',allow_pickle=True)
lsmc = convert_array_tensor(lsmc)
lsmc = np.random.choice(lsmc,len(cnn))


lsmc_test = np.load('Results/Energy/LSMC_test_'+str(N)+'.npy',allow_pickle=True)
lsmc_test = convert_array_tensor(lsmc_test)
lsmc_test = np.random.choice(lsmc_test,len(cnn))


#plt.figure(figsize=(15,5))
sns.displot({'LSMC Train Vol.':lsmc,'LSMC Test Vol.':lsmc_test,'Max Value Dist.':max_dist,'CNN':cnn}, kind="kde", fill=True,legend=True).set_axis_labels('Average Payoff','Density')
#sns.displot(cnn, kind="kde", fill=True,label = 'CNN')
plt.title('Payoff comprarison| LSCM x CNN x Max| N = '+str(N))
plt.show()

#################################################################################
# Crude oil price plotting -GARCH
#################################################################################

N =  30

cnn = np.load('Results/Energy/Becker_cnn_'+str(N)+'.npy',allow_pickle=True)
cnn = convert_array_tensor(cnn)

max_dist = np.load('Results/Energy/Max_dist_'+str(N)+'.npy',allow_pickle=True)
max_dist = convert_array_tensor(max_dist)


lsmc_test = np.load('Results/Energy/LSMC_garch_'+str(N)+'.npy',allow_pickle=True)
lsmc_test = convert_array_tensor(lsmc_test)
lsmc_test = np.random.choice(lsmc_test,len(cnn))


#plt.figure(figsize=(15,5))
sns.displot({'LSMC GARCH Vol.':lsmc_test,'Max Value Dist.':max_dist,'CNN':cnn}, kind="kde", fill=True,legend=True).set_axis_labels('Average Payoff','Density')
#sns.displot(cnn, kind="kde", fill=True,label = 'CNN')
plt.title('Payoff comprarison| LSCM x CNN x Max| N = '+str(N))
plt.show()

#################################################################################
# Crude oil price plotting - testing data for LSMC
#################################################################################

N =  30

cnn = np.load('Results/Energy/Becker_cnn_'+str(N)+'.npy',allow_pickle=True)
cnn = convert_array_tensor(cnn)

max_dist = np.load('Results/Energy/Max_dist_'+str(N)+'.npy',allow_pickle=True)
max_dist = convert_array_tensor(max_dist)


lsmc_test = np.load('Results/Energy/LSMC_apply_test_'+str(N)+'.npy',allow_pickle=True)
lsmc_test = convert_array_tensor(lsmc_test)
lsmc_test = np.random.choice(lsmc_test,len(cnn))


#plt.figure(figsize=(15,5))
sns.displot({'LSMC testing returns Vol.':lsmc_test,'Max Value Dist.':max_dist,'CNN':cnn}, kind="kde", fill=True,legend=True).set_axis_labels('Average Payoff','Density')
#sns.displot(cnn, kind="kde", fill=True,label = 'CNN')
plt.title('Payoff comprarison| LSCM x CNN x Max| N = '+str(N))
plt.xlim(-5,20)
plt.show()


#################################################################################
# Crude oil price plotting - training data for CNN pricing
#################################################################################

N =  30

cnn = np.load('Results/Energy/Becker_cnn_data_train_'+str(N)+'.npy',allow_pickle=True)
cnn = convert_array_tensor(cnn)

max_dist = np.load('Results/Energy/Max_dist_data_train_'+str(N)+'.npy',allow_pickle=True)
max_dist = convert_array_tensor(max_dist)


lsmc_test = np.load('Results/Energy/LSMC_GARCH_'+str(N)+'.npy',allow_pickle=True)
lsmc_test = convert_array_tensor(lsmc_test)
lsmc_test = np.random.choice(lsmc_test,len(cnn))


#plt.figure(figsize=(15,5))
sns.displot({'LSMC testing returns Vol.':lsmc_test,'Max Value Dist.':max_dist,'CNN':cnn}, kind="kde", fill=True,legend=True).set_axis_labels('Average Payoff','Density')
#sns.displot(cnn, kind="kde", fill=True,label = 'CNN')
plt.title('Payoff comprarison| LSCM x CNN x Max| N = '+str(N))
plt.xlim(-5,20)
plt.show()






"""
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


"""


