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
            df_temp = pd.DataFrame({'Models':[model]*len(payoffs),'N':[N]*len(payoffs),'Expected Payoff':payoffs})
            df_results = df_results.append(df_temp).reset_index().drop(columns = ['index'])
            df_results['Models'].replace('BeckeH','Becker', inplace=True)
            df_results['Models'].replace('Becker_cnn','CNN', inplace=True)
            #df_results['Models'].replace('Max_dist','Max Avg Payoff', inplace=True)
    return df_results



#############################################################################
# Plotting the Avg Pay with standar deviation for each N on Brownian motion:
############################################################################# 


path = 'Results/BROWNIAN/'
file_name = 'table_beckerdm.txt'
model_names = 'BeckeH', 'Becker_cnn','Max_dist'

results = pd.read_csv(path+file_name, delimiter = ";", header=None)
results.columns = ['N','Payoff','Max Payoff','Payoff Std']

N_series = results.N.values.astype(str)

df_for_seaborn = create_dataframe_two_models(path,model_names,N_series)

df_for_seaborn = df_for_seaborn[df_for_seaborn['Models']!='Max_dist']

f = plt.figure(figsize=(6,3))
g =  sns.lineplot(data=df_for_seaborn[['Models','N','Expected Payoff']], x="N", y="Expected Payoff", hue="Models",style='Models',palette = 'binary',ci = 95)
g.set_xticks(np.arange(0, len(N_series)/2, 2))
f.savefig('brownian_results.pdf', bbox_inches='tight')

#################################################################################
# Plotting the Avg Pay with standar deviation for each N on Harmonic time-series:
################################################################################# 


path = 'Results/HARM/'
file_name = 'harmonic_results.txt'
model_names = 'BeckeH', 'Becker_cnn'

results = pd.read_csv(path+file_name, delimiter = ";", header=None)
results.columns = ['N','Payoff','Max Payoff','Payoff Std']

N_series = np.unique(results.N.values).astype(str)

df_for_seaborn = create_dataframe_two_models(path,model_names,N_series)

temp = df_for_seaborn.loc[df_for_seaborn['Models']=='Becker']
temp.loc[temp['N']=='10'].std()
temp.loc[temp['N']=='10'].mean()

f = plt.figure(figsize=(15,5))
sns.lineplot(data=df_for_seaborn, x="N", y="Average Payoff", hue="Models",style='Models',palette = 'binary',ci=None)
#sns.lineplot(data=df_for_seaborn, x="N", y="Payoff", hue="Models",palette = 'binary')
# pallet options: Set1
f.savefig('harmonic_results.pdf', bbox_inches='tight')


########################################################################################################
# Plotting the Avg Pay with standar deviation for each N on fractional Brownian motion time-series:
######################################################################################################## 



path = 'Results/FBM/'
file_name = 'table_beckerfbm.txt'
model_names = 'BeckeH', 'Becker_cnn'

results = pd.read_csv(path+file_name, delimiter = ";", header=None)
results.columns = ['N','Payoff','Max Payoff','Payoff Std']

N_series = np.unique(results.N.values).astype(str)

df_for_seaborn = create_dataframe_two_models(path,model_names,N_series)

temp = df_for_seaborn.loc[df_for_seaborn['Models']=='Becker']
temp.loc[temp['N']=='10'].std()
temp.loc[temp['N']=='10'].mean()

f = plt.figure(figsize=(15,5))
sns.lineplot(data=df_for_seaborn, x="N", y="Average Payoff", hue="Models",style='Models',palette = 'binary')

f.savefig('fbm_results.pdf', bbox_inches='tight')


#################################################################################
# Plotting the average payoff on each decision point
#################################################################################


path = 'Results/HARM/'
file_name = 'harmonic_results.txt'

model_names = 'BeckeH', 'Becker_cnn'

results = pd.read_csv(path+file_name, delimiter = ";", header=None)
results.columns = ['N','Payoff','Max Payoff','Payoff Std']

N_series = np.unique(results.N.values.astype(str))

models = ['becker','cnn']

model_name = models[1] # We use the first value for test

N_test = ['190','280']

for N in N_test:
    avg_payoff = np.squeeze(np.load(path+'exerc_region_N_'+N+'_'+model_name+'.npy'),0)
    time_steps = np.arange(0,len(avg_payoff))
    
    MA_param = 3
    avg_payoff = moving_average(avg_payoff, MA_param)
    #avg_payoff = np.flip(avg_payoff)
    time_steps = time_steps[:-MA_param+1]
    
    
    
    plt.plot(time_steps, avg_payoff, lw=1,color = 'gray')
    d = np.zeros(len(avg_payoff))
    #d = np.ones(len(avg_payoff))*max(avg_payoff)
    plt.fill_between(time_steps, avg_payoff, where=avg_payoff>=d, interpolate=True, color='gray', alpha=0.30)
    plt.show()
# ax.fill_between(time_steps, avg_payoff, where=avg_payoff>=d, interpolate=True, color='blue')
# ax.fill_between(time_steps, avg_payoff, where=avg_payoff<=d, interpolate=True, color='red')


#################################################################################
# Plotting the training evolution for harmonic time-series
#################################################################################

results = pd.read_csv(path+file_name, delimiter = ";", header=None)
results.columns = ['N','Payoff','Max Payoff','Payoff Std']

N_series = results.N.values.astype(str)

models = ['Becker_cnn_train_','Becker_train_']

N = N_series[30] # We use the first value for test

N_test = ['100','280','490']

MA_param = 50
d = np.zeros(len(avg_payoff))
fig = plt.figure(figsize=(15,5))
t = 2
for N in N_test:
    avg_payoff_cnn = np.load(path + 'Becker_cnn_train_'+N+'.npy')
    avg_payoff_becker = np.load(path + 'Becker_train_'+N+'.npy')
    time_steps = np.arange(0,len(avg_payoff_cnn))
    t = t-2.0/len(N_test)
    plt.plot(time_steps[:-MA_param+1], moving_average(avg_payoff_cnn,MA_param),color = (t/2.0, t/2.0, t/2.0),linestyle='--',label = N+' CNN')
    plt.plot(time_steps[:-MA_param+1], moving_average(avg_payoff_becker,MA_param),color = (t/2.0, t/2.0, t/2.0),label = N+' Becker')
plt.legend(loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Average Payoff')
plt.savefig('training_harmonic.pdf', bbox_inches='tight')

#################################################################################
# Plotting the training evolution for FBM
#################################################################################

path = 'Results/FBM/'
file_name = 'table_beckerfbm.txt'


results = pd.read_csv(path+file_name, delimiter = ";", header=None)
results.columns = ['N','Payoff','Max Payoff','Payoff Std']

N_series = results.N.values.astype(str)

models = ['Becker_cnn_train_','Becker_train_']
N_test = ['100','280','490']
#N_test = ['280']

MA_param = 50
#d = np.zeros(len(avg_payoff))
fig = plt.figure(figsize=(15,5))
t = 2
for N in N_test:
    avg_payoff_cnn = np.load(path + 'Becker_cnn_train_'+N+'.npy')
    avg_payoff_becker = np.load(path + 'Becker_train_'+N+'.npy')
    time_steps = np.arange(0,len(avg_payoff_cnn))
    t = t-2.0/len(N_test)
    plt.plot(time_steps[:-MA_param+1], moving_average(avg_payoff_cnn,MA_param),color = (t/2.0, t/2.0, t/2.0),linestyle='--',label = N+' CNN')
    plt.plot(time_steps[:-MA_param+1], moving_average(avg_payoff_becker,MA_param),color = (t/2.0, t/2.0, t/2.0),label = N+' Becker')
plt.legend(loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Average Payoff')
plt.savefig('training_fbm.pdf', bbox_inches='tight')


########################################################################################################
# Plotting the Avg Pay with standard deviation for each N real-word data (crude oil)
######################################################################################################## 



path = 'Results/Energy/'
file_name = 'table_cnnreal_val.txt'
model_names = 'Becker_cnn','BeckeH', 'LSMC','Max_dist'

results = pd.read_csv(path+file_name, delimiter = ";", header=None)
results.columns = ['N','Payoff','Max Payoff','Payoff Std']

N_series = np.unique(results.N.values).astype(str)[:-1]

df_for_seaborn = create_dataframe_two_models(path,model_names,N_series)     

f = plt.figure(figsize=(6,3))
g = sns.lineplot(data=df_for_seaborn, x="N", y="Expected Payoff", hue="Models",style='Models',palette = 'binary',ci='sd')
g.set_xticks(np.arange(0, len(N_series)/2, 2)) 
g.legend(title='Models', loc='upper left', labels=['CNN','Becker','LSMC','Max Avg Payoff']) 
#sns.lineplot(data=df_for_seaborn, x="N", y="Payoff", hue="Models",palette = 'binary')
# pallet options: Set1
f.savefig('energy_results.pdf', bbox_inches='tight')



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

