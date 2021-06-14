# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 05:45:49 2021

@author: leona
Testing
"""

import torch
from torch import nn
import numpy as np
import pandas as pd
from LSTM import Neural_Net_LSTM
from MLP import Neural_Net_NN
from CNN import Neural_Net_CNN
from utils import choose_gen,torch_t,select_input,calculate_avg_payoff,write_file,write_table,generate_avg_payoff_step,save_exer_region

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')


    
    
def Becker_test_model(s_0,K,T,N,r,delta,sigma,d,batch_size,order,type_of_data,
                     PATH,num_neurons,lr_boundaries_NN,lr_init,mc_runs,train_steps,file,path,path_output):
    print('\n\n Evaluation phase by Becker´s method:\n\n')
    px_vec = []
    px_vec_max = []
    
    neural_net = Neural_Net_NN(num_neurons,d+1).to(device)
    neural_net.load_state_dict(torch.load(path_output+'/best_model_becker.pt'))
    neural_net.eval()  
    
    gen_func = choose_gen(type_of_data)
    
    neural_net.eval()  
    for mc_step in range(0,mc_runs+1):
        #Generate batch data
        seed = mc_step+train_steps+1
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        input_gen = select_input(s_0,K,T,N,r,delta,sigma,d,batch_size,seed,type_of_data,order,'none',file,path)
        
        X,p_,g_tau = gen_func(input_gen)
        
        X,p_,g_tau = torch_t(X), torch_t(p_),torch_t(g_tau)
        state = torch.cat((X,p_),axis = 1)
    
        for n in range(N-2, -1, -1): # loop from T-T/N to T/N
            net_n = neural_net(state[:,:,n])
            g_tau = torch.where(net_n > 0, p_[:, :, n], g_tau) # this is our new g_tau now, we hand it backwards in time
        
        px_mean_batch = torch.mean(g_tau)
        if px_mean_batch.item() < torch.mean(torch.max(p_,2)[0]).item():
            px_vec.append(px_mean_batch)
            px_vec_max.append(torch.mean(torch.max(p_,2)[0]))
        if mc_step%100 == 0 or mc_step <100:
            print('| Mc run step: {:5.0f} | Avg price: {:3.3f} |'.format(mc_step,px_mean_batch.item()))
    px_mean_max = torch.mean(torch.stack(px_vec_max))
    px_mean = torch.mean(torch.stack(px_vec))
    px_std = torch.std(torch.stack(px_vec))
    text = '\n Params: N = '+str(N)+' | d = '+str(d)+' | type of data: '+type_of_data+'| Becker Avg price = '+ str(px_mean.item())+' | Max Price: '+str(px_mean_max.item())
    write_file(text)
    write_table('becker',N,px_mean.item(),px_mean_max.item(),px_std.item())
    del neural_net
    torch.cuda.empty_cache()

def BeckeH_r_test_model(s_0,K,T,N,r,delta,sigma,d,batch_size,order,type_of_data,
                     PATH,num_neurons,lr_boundaries_NN,lr_init,mc_runs,train_steps,file,path,path_output):
    print('\n\n Evaluation phase by Becker´s method:\n\n')
    px_vec = []
    px_vec_max = []
    
    neural_net = Neural_Net_NN(num_neurons,(d+1)*N).to(device)
    
    neural_net.load_state_dict(torch.load(path_output+'/best_model_becker.pt'))
    neural_net.train()  
    
    gen_func = choose_gen(type_of_data)
    
    neural_net.eval()  
    for mc_step in range(0,mc_runs+1):
        #Generate batch data
        seed = mc_step+train_steps+1
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        input_gen = select_input(s_0,K,T,N,r,delta,sigma,d,batch_size,seed,type_of_data,order,'none',file,path)
        
        X,p_,g_tau = gen_func(input_gen)
        
        X,p_,g_tau = torch_t(X), torch_t(p_),torch_t(g_tau)
        X = torch.diff(X,1,2,prepend = torch.unsqueeze(X[:,:,0],2))
        state = torch.cat((X,p_),axis = 1)
        
    
        state_hist = []
        hist_state = torch.zeros((state.shape[0],state.shape[1],N)).to(device)
        for n in range(0, N, 1):
            hist_state = torch.cat((hist_state[:,:,1:],torch.unsqueeze(state[:,:,n], dim = 2)), dim=2)
            state_hist.append(hist_state.view(state.shape[0],-1))
        
        for n in range(N-2, -1, -1): # loop from T-T/N to T/N
            net_n = neural_net(state_hist[n])
            g_tau = torch.where(net_n > 0, p_[:, :, n], g_tau) # this is our new g_tau now, we hand it backwards in time
        
        px_mean_batch = torch.mean(g_tau)
        if px_mean_batch.item() < torch.mean(torch.max(p_,2)[0]).item():
            px_vec.append(px_mean_batch)
            px_vec_max.append(torch.mean(torch.max(p_,2)[0]))
        
        
        if mc_step%100 == 0 or mc_step <100:
            print('| Mc run step: {:5.0f} | Avg price: {:3.3f} |'.format(mc_step,px_mean_batch.item()))
    px_mean_max = torch.mean(torch.stack(px_vec_max))
    px_mean = torch.mean(torch.stack(px_vec))
    px_std = torch.std(torch.stack(px_vec))
    text = '\n Params: N = '+str(N)+' | d = '+str(d)+' | type of data: '+type_of_data+'| Becker Hist, returns Avg price = '+ str(px_mean.item())+' | Max Price: '+str(px_mean_max.item())
    write_file(text)
    write_table('becker',N,px_mean.item(),px_mean_max.item(),px_std.item())
    np.save(path_output+'BeckeH_r_'+str(N),np.asarray(px_vec))
    del neural_net
    torch.cuda.empty_cache()
    
def BeckeH_test_model(s_0,K,T,N,r,delta,sigma,d,batch_size,order,type_of_data,
                     PATH,num_neurons,lr_boundaries_NN,lr_init,mc_runs,train_steps,file,path,path_output):
    print('\n\n Evaluation phase by Becker´s method:\n\n')
    px_vec = []
    px_vec_max = []
    df_posi_payoff = pd.DataFrame(columns = ['position','payoffs'])
    
    neural_net = Neural_Net_NN(num_neurons,(d+1)*N).to(device)
    
    neural_net.load_state_dict(torch.load(path_output+'/best_model_becker.pt'))
    neural_net.train()  
    
    gen_func = choose_gen(type_of_data)
    
    neural_net.eval()  
    for mc_step in range(0,mc_runs+1):
        #Generate batch data
        output_hist=[]
        seed = mc_step+train_steps+1
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        input_gen = select_input(s_0,K,T,N,r,delta,sigma,d,batch_size,seed,type_of_data,order,'none',file,path)
        
        X,p_,g_tau = gen_func(input_gen)
        
        X,p_,g_tau = torch_t(X), torch_t(p_),torch_t(g_tau)
        state = torch.cat((X,p_),axis = 1)
    
    
        state_hist = []
        hist_state = torch.zeros((state.shape[0],state.shape[1],N)).to(device)
        for n in range(0, N, 1):
            hist_state = torch.cat((hist_state[:,:,1:],torch.unsqueeze(state[:,:,n], dim = 2)), dim=2)
            state_hist.append(hist_state.view(state.shape[0],-1))
        
        for n in range(N-2, -1, -1): # loop from T-T/N to T/N
            net_n = neural_net(state_hist[n])
            g_tau = torch.where(net_n > 0, p_[:, :, n], g_tau) # this is our new g_tau now, we hand it backwards in time
        
        px_mean_batch = torch.mean(g_tau)
        posi_payoff = generate_avg_payoff_step(output_hist,p_,batch_size)
        df_posi_payoff = df_posi_payoff.append(posi_payoff)
        
        if px_mean_batch.item() < torch.mean(torch.max(p_,2)[0]).item():
            px_vec.append(px_mean_batch)
            px_vec_max.append(torch.mean(torch.max(p_,2)[0]))
        
        if mc_step%100 == 0 or mc_step <100:
            print('| Mc run step: {:5.0f} | Avg price: {:3.3f} |'.format(mc_step,px_mean_batch.item()))
    px_mean_max = torch.mean(torch.stack(px_vec_max))
    px_mean = torch.mean(torch.stack(px_vec))
    px_std = torch.std(torch.stack(px_vec))
    save_exer_region(df_posi_payoff,N,'becker',path_output)
    text = '\n Params: N = '+str(N)+' | d = '+str(d)+' | type of data: '+type_of_data+'| Becker Hist Avg price = '+ str(px_mean.item())+' | Max Price: '+str(px_mean_max.item())
    write_file(text)
    write_table('becker',N,px_mean.item(),px_mean_max.item(),px_std.item())
    np.save(path_output+'BeckeH_'+str(N),np.asarray(px_vec))
    del neural_net
    torch.cuda.empty_cache()
    



def Becker_mod_cnn_test_model(s_0,K,T,N,r,delta,sigma,d,batch_size,order,type_of_data,
                     PATH,num_neurons,lr_boundaries_NN,lr_init,mc_runs,train_steps,file,path,path_output):
    print('\n\n Evaluation phase by Becker´s method:\n\n')
    px_vec = []
    px_vec_max = []
    
    df_posi_payoff = pd.DataFrame(columns = ['position','payoffs'])
    neural_net = Neural_Net_CNN(num_neurons,d+1,N,2,batch_size).to(device)
    
    neural_net.load_state_dict(torch.load(path_output+'/best_model_becker.pt'))
    neural_net.eval()  
    
    gen_func = choose_gen(type_of_data)
    
    K_var = torch.ones(batch_size,1).to(device)*K
    T_var = torch.ones(batch_size,1).to(device)*T
    non_tran_var = torch.cat((K_var,T_var),1)
    
    neural_net.eval()  
    for mc_step in range(0,mc_runs+1):
        #Generate batch data
        output_hist = []
        seed = mc_step+train_steps+1
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        input_gen = select_input(s_0,K,T,N,r,delta,sigma,d,batch_size,seed,type_of_data,order,'test',file,path)
        
        X,p_,g_tau = gen_func(input_gen)
        
        X,p_,g_tau = torch_t(X), torch_t(p_),torch_t(g_tau)
        X = torch.diff(X,1,2,prepend = torch.unsqueeze(X[:,:,0],2))
        state = torch.cat((X,p_),axis = 1)
        
        
        loss = np.zeros(1)
        loss = torch_t(loss)
        state_hist = []
        hist_state = torch.zeros((state.shape[0],state.shape[1],N)).to(device)
        for n in range(0, N, 1):
            hist_state = torch.cat((hist_state[:,:,1:],torch.unsqueeze(state[:,:,n], dim = 2)), dim=2)
            state_hist.append(torch.unsqueeze(hist_state, dim=3))
        state_hist = torch.cat(state_hist,3)

        for n in range(N-2, -1, -1): # loop from T-T/N to T/N
            with torch.no_grad():    
                net_n = neural_net(state_hist[:,:,:,n],non_tran_var)
                output_hist.append(net_n)
            g_tau = torch.where(net_n > 0, p_[:, :, n], g_tau) # this is our new g_tau now, we hand it backwards in time
        
        px_mean_batch = torch.mean(g_tau)
        
        posi_payoff = generate_avg_payoff_step(output_hist,p_,batch_size)
        df_posi_payoff = df_posi_payoff.append(posi_payoff)
        
        if px_mean_batch.item() < torch.mean(torch.max(p_,2)[0]).item():
            px_vec.append(px_mean_batch)
            px_vec_max.append(torch.mean(torch.max(p_,2)[0]))
        
        if mc_step%100 == 0 or mc_step <100:
            print('| Mc run step: {:5.0f} | Avg price: {:3.3f} |'.format(mc_step,px_mean_batch.item()))
    px_mean_max = torch.mean(torch.stack(px_vec_max))
    px_mean = torch.mean(torch.stack(px_vec))
    px_std = torch.std(torch.stack(px_vec))
    save_exer_region(df_posi_payoff,N,'cnn',path_output)
    
    text = '\n Params: N = '+str(N)+' | d = '+str(d)+' | type of data: '+type_of_data+'| Becker/Ours CNN Avg price = '+ str(px_mean.item())+' | Max Price: '+str(px_mean_max.item())
    write_file(text)
    write_table('cnn',N,px_mean.item(),px_mean_max.item(),px_std.item())
    np.save(path_output+'Becker_cnn_'+str(N),np.asarray(px_vec))
    np.save(path_output+'Max_dist_'+str(N),np.asarray(px_vec_max))
    del neural_net
    torch.cuda.empty_cache()