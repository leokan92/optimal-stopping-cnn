# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 04:29:52 2021

@author: leona

utils functions
"""
import torch
import numpy as np


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
    return gen_function

def select_input(s_0,K,T,N,r,delta,sigma,d,batch_size,seed,type_of_data,order,path,file):
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
    return selected_inputs

def write_file(text):
    f = open("results.txt", "a")
    f.write(text)
    f.close()
    
def write_table(N,P,max_P,std_P):
    f = open("table.txt", "a")
    text = str(N)+';'+str(P)+';'+str(max_P)+';'+str(std_P)+'\n'
    f.write(text)
    f.close()