# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 04:29:52 2021

@author: leona

utils functions
"""
import torch
import numpy as np
import pandas as pd


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')


def torch_t(x): return torch.from_numpy(x).float().to(device)


def calculate_avg_payoff(net_n_list,flip,p_):   
    a = np.squeeze(torch.stack(net_n_list,axis = 1).cpu().detach().numpy(),axis = 2)     
    b = np.where(a>0,True,False)
    if flip:
        c = np.flip(b,axis=1)
        posi = np.argmax(c,axis=1)
    else:
        posi = np.argmax(b,axis=1)
    p_1  = np.squeeze(p_,axis=1)
    return torch.mean(p_1[range(0,len(p_1)),posi])
        
def choose_gen(type_of_data):
    if type_of_data == 'dm':
        from gen_data import generate_data_normal_dist_bermudan
        gen_function = generate_data_normal_dist_bermudan
    elif type_of_data == 'ar':
        from gen_data import generate_data_ar_model
        gen_function = generate_data_ar_model
    if type_of_data == 'narma':
        from gen_data import generate_data_narma_model
        gen_function = generate_data_narma_model
    if type_of_data == 'fbm':
        from gen_data import generate_data_fractional_brownian_bermudan
        gen_function = generate_data_fractional_brownian_bermudan
    if type_of_data == 'heston':
        from gen_data import generate_data_heston_model
        gen_function = generate_data_heston_model
    if type_of_data == 'harmonic':
        from gen_data import generate_data_harmionic_model
        gen_function = generate_data_harmionic_model
    if type_of_data == 'real':
        from gen_data import generate_real_data_sample
        gen_function = generate_real_data_sample
    if type_of_data == 'real_val':
        from gen_data import generate_real_data_sample_val
        gen_function = generate_real_data_sample_val
    return gen_function

def select_input(s_0,K,T,N,r,delta,sigma,d,batch_size,seed,type_of_data,order,sample_type,path,file):
    if type_of_data == 'dm':
        selected_inputs = (s_0,K,T,N,r,delta,sigma,d,seed,batch_size)
    elif type_of_data == 'ar':
        selected_inputs = (s_0,K,T,N,r,d,batch_size)
    if type_of_data == 'narma':
        selected_inputs = (s_0,K,T,N,r,d,batch_size,order)
    elif type_of_data == 'fbm':
        selected_inputs = (s_0,K,T,N,r,d,batch_size,seed)
    if type_of_data == 'heston':
        selected_inputs = (seed,N,d,s_0,K,sigma,0.1,-0.2,2,0.04,0.3,T,r,batch_size,False)
    elif type_of_data == 'harmonic':
        selected_inputs = (s_0,K,T,N,r,d,batch_size)
    if type_of_data =='real':
        selected_inputs = (path,file,N,d,batch_size,s_0,T,r,K)
    if type_of_data =='real_val':
        selected_inputs = (path,file,N,d,batch_size,s_0,T,r,K,sample_type)
    return selected_inputs

def write_file(text):
    f = open("results.txt", "a")
    f.write(text)
    f.close()
    
def write_table(model_name,N,P,max_P,std_P):
    f = open('table+'+model_name+'.txt', 'a')
    text = str(N)+';'+str(P)+';'+str(max_P)+';'+str(std_P)+'\n'
    f.write(text)
    f.close()
    
def calculate_sigma(path,file):
    df = pd.read_csv(path+file,sep=';',thousands=',')
    #returns = df['Close']
    #returns = np.diff(df['Close'])  
    #returns = np.diff(df['Close'])/df['Close'][1:]
    returns = np.std(df['Close'])/np.mean(df['Close'])
    #return np.std(df['Close'])
    return returns

def generate_avg_payoff_step(output_hist,p_,batch_size):
    output_hist.insert(0,torch.ones((batch_size,1)).to(device))
    array_outputs = torch.squeeze(torch.stack(output_hist),2).cpu().detach().numpy()
    positions = np.argmax(np.where(np.flip(array_outputs,0)>0,True,False),0)
    payoff = torch.squeeze(p_,1).cpu().detach().numpy()
    df_posi_payoff = pd.DataFrame({'position':positions,'payoffs':payoff[range(0,len(payoff)),positions]})
    return df_posi_payoff

def save_exer_region(df_posi_payoff,N,model_name,path_output):
    diff = np.setdiff1d(np.arange(N),(df_posi_payoff.position.unique()))
    zeros = np.zeros(len(np.setdiff1d(np.arange(N),(df_posi_payoff.position.unique()))))
    df_posi_payoff_zeros = pd.DataFrame({'position':diff,'payoffs':zeros})
    df_posi_payoff = df_posi_payoff.append(df_posi_payoff_zeros)
    np.save(path_output+'exerc_region_N_'+str(N)+'_'+model_name+'.npy',df_posi_payoff.groupby('position').mean().to_numpy().T)