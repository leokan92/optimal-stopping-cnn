# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 04:56:17 2021

@author: leona

Train the LSTM_model
"""
import torch
from torch import nn
import numpy as np
import pandas as pd
from LSTM import Neural_Net_LSTM
from MLP import Neural_Net_NN
from CNN import Neural_Net_CNN
from utils import choose_gen,torch_t,select_input,calculate_avg_payoff,generate_avg_payoff_step
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')



def Becker_train_model(s_0,K,T,N,r,delta,sigma,d,batch_size,order,type_of_data,
                     PATH,num_neurons,lr_boundaries,lr_init,training_steps,file,path):
    
    neural_net = Neural_Net_NN(num_neurons,d+1).to(device)
    optimizer = torch.optim.Adam(neural_net.parameters(), lr=0.05)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_boundaries, gamma=0.1)
    
    gen_func = choose_gen(type_of_data)
    px_hist = []
    print('\n\n Training phase:\n\n')
    max_px = 0
    for train_step in range(0,training_steps+1):
        #Generate batch data
        seed = train_step
        torch.manual_seed(seed)
        np.random.seed(seed)

        input_gen = select_input(s_0,K,T,N,r,delta,sigma,d,batch_size,seed,type_of_data,order,'none',file,path)
        X,p_,g_tau = gen_func(input_gen)

        X,p_,g_tau = torch_t(X), torch_t(p_),torch_t(g_tau)
        state = torch.cat((X,p_),axis = 1)
        
        loss = np.zeros(1)
        loss = torch_t(loss)
        for n in range(N-2, -1, -1): # loop from T-T/N to T/N
            net_n = neural_net(state[:,:,n])
            F_n   = torch.sigmoid(net_n)
            loss -= torch.mean(p_[:, :, n] * F_n + g_tau * (1. - F_n)) # the loss for a single stopping time problem, we want to maximize this loss in every time step
            g_tau = torch.where(net_n > 0, p_[:, :, n], g_tau) # this is our new g_tau now, we hand it backwards in time
        
        
        px_mean_batch = torch.mean(g_tau)
        loss = torch.mean(loss)
        px_hist.append(px_mean_batch.item())
        
        if train_step> 10:
            if px_mean_batch.item()>max_px and px_mean_batch.item()<np.mean(px_hist[-10:-1])*1.30:
            #if np.mean(px_hist[-100:])>np.mean(px_hist[-500:]):
                #print('Best Model Saved')
                torch.save(neural_net.state_dict(), PATH+'/best_model_becker.pt')
        
        
        if train_step%100 == 0 or train_step <100:
            print('| Train step: {:5.0f} | Loss: {:3.3f} | Avg price: {:3.3f} | Lr: {:1.6f} |'.format(train_step,loss.item(),px_mean_batch.item(),optimizer.param_groups[0]['lr']))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

def BeckerH_r_train_model(s_0,K,T,N,r,delta,sigma,d,batch_size,order,type_of_data,
                     PATH,num_neurons,lr_boundaries,lr_init,training_steps,file,path):
    
    neural_net = Neural_Net_NN(num_neurons,(d+1)*N).to(device)
    optimizer = torch.optim.Adam(neural_net.parameters(), lr=0.05)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_boundaries, gamma=0.1)
    
    gen_func = choose_gen(type_of_data)
    px_hist = []
    print('\n\n Training phase:\n\n')
    max_px = 0
    for train_step in range(0,training_steps+1):
        #Generate batch data
        seed = train_step
        torch.manual_seed(seed)
        np.random.seed(seed)

        input_gen = select_input(s_0,K,T,N,r,delta,sigma,d,batch_size,seed,type_of_data,order,'none',file,path)
        X,p_,g_tau = gen_func(input_gen)

        X,p_,g_tau = torch_t(X), torch_t(p_),torch_t(g_tau)
        X = torch.diff(X,1,2,prepend = torch.unsqueeze(X[:,:,0],2))
        state = torch.cat((X,p_),axis = 1)
        
        
        loss = np.zeros(1)
        loss = torch_t(loss)
        state_hist = []
        hist_state = torch.zeros((state.shape[0],state.shape[1],N)).to(device)
        for n in range(0, N, 1):
            state_for_append = torch.cat((hist_state[:,:,n+1:],torch.flip(state[:,:,:n+1],dims=[0,2])), dim=2)
            state_hist.append(state_for_append.view(state.shape[0],-1))
        
        for n in range(N-2, -1, -1): # loop from T-T/N to T/N
            net_n = neural_net(state_hist[n])
            F_n   = torch.sigmoid(net_n)
            loss -= torch.mean(p_[:, :, n] * F_n + g_tau * (1. - F_n)) # the loss for a single stopping time problem, we want to maximize this loss in every time step
            g_tau = torch.where(net_n > 0, p_[:, :, n], g_tau) # this is our new g_tau now, we hand it backwards in time
        
        
        px_mean_batch = torch.mean(g_tau)
        loss = torch.mean(loss)
        px_hist.append(px_mean_batch.item())
        
        if train_step> 10:
            if px_mean_batch.item()>max_px and px_mean_batch.item()<np.mean(px_hist[-10:-1])*1.30:
            #if np.mean(px_hist[-100:])>np.mean(px_hist[-500:]):
                #print('Best Model Saved')
                torch.save(neural_net.state_dict(), PATH+'/best_model_becker.pt')
        
        
        if train_step%100 == 0 or train_step <100:
            print('| Train step: {:5.0f} | Loss: {:3.3f} | Avg price: {:3.3f} | Lr: {:1.6f} |'.format(train_step,loss.item(),px_mean_batch.item(),optimizer.param_groups[0]['lr']))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        
def BeckerH_train_model(s_0,K,T,N,r,delta,sigma,d,batch_size,order,type_of_data,
                     PATH,num_neurons,lr_boundaries,lr_init,training_steps,file,path,path_output):
    
    neural_net = Neural_Net_NN(num_neurons,(d+1)*N).to(device)
    optimizer = torch.optim.Adam(neural_net.parameters(), lr=0.05)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_boundaries, gamma=0.1)
    
    gen_func = choose_gen(type_of_data)
    px_hist = []
    print('\n\n Training phase:\n\n')
    max_px = 0
    for train_step in range(0,training_steps+1):
        #Generate batch data
        seed = train_step
        torch.manual_seed(seed)
        np.random.seed(seed)

        input_gen = select_input(s_0,K,T,N,r,delta,sigma,d,batch_size,seed,type_of_data,order,'none',file,path)
        X,p_,g_tau = gen_func(input_gen)

        X,p_,g_tau = torch_t(X), torch_t(p_),torch_t(g_tau)
        state = torch.cat((X,p_),axis = 1)
        
        loss = np.zeros(1)
        loss = torch_t(loss)
        state_hist = []
        hist_state = torch.zeros((state.shape[0],state.shape[1],N)).to(device)
        for n in range(0, N, 1):
            state_for_append = torch.cat((hist_state[:,:,n+1:],torch.flip(state[:,:,:n+1],dims=[0,2])), dim=2)
            state_hist.append(state_for_append.view(state.shape[0],-1))
        
        for n in range(N-2, -1, -1): # loop from T-T/N to T/N
            net_n = neural_net(state_hist[n])
            F_n   = torch.sigmoid(net_n)
            loss -= torch.mean(p_[:, :, n] * F_n + g_tau * (1. - F_n)) # the loss for a single stopping time problem, we want to maximize this loss in every time step
            g_tau = torch.where(net_n > 0, p_[:, :, n], g_tau) # this is our new g_tau now, we hand it backwards in time
        
        
        px_mean_batch = torch.mean(g_tau)
        loss = torch.mean(loss)
        px_hist.append(px_mean_batch.item())
        
        if train_step> 10:
            if px_mean_batch.item()>max_px and px_mean_batch.item()<np.mean(px_hist[-10:-1])*1.30:
            #if np.mean(px_hist[-100:])>np.mean(px_hist[-500:]):
                #print('Best Model Saved')
                torch.save(neural_net.state_dict(), PATH+'/best_model_becker.pt')
        
        
        if train_step%100 == 0 or train_step <100:
            print('| Train step: {:5.0f} | Loss: {:3.3f} | Avg price: {:3.3f} | Lr: {:1.6f} |'.format(train_step,loss.item(),px_mean_batch.item(),optimizer.param_groups[0]['lr']))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    np.save(path_output+'Becker_train_'+str(N),np.asarray(px_hist))


      
def Becker_mod_cnn_train_model(s_0,K,T,N,r,delta,sigma,d,batch_size,order,type_of_data,
                     PATH,num_neurons,lr_boundaries,lr_init,training_steps,file,path,path_output):
    
    
    neural_net = Neural_Net_CNN(num_neurons,d+1,N,2,batch_size).to(device)
    optimizer = torch.optim.Adam(neural_net.parameters(), lr=0.05)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_boundaries, gamma=0.1)
    
    gen_func = choose_gen(type_of_data)
    
    K_var = torch.ones(batch_size,1).to(device)*K
    T_var = torch.ones(batch_size,1).to(device)*T
    
    non_tran_var = torch.cat((K_var,T_var),1)
    px_hist = []
    max_px = 0
    print('\n\n Training phase:\n\n')
    for train_step in range(0,training_steps+1):
        #Generate batch data
        seed = train_step
        torch.manual_seed(seed)
        np.random.seed(seed)

        input_gen = select_input(s_0,K,T,N,r,delta,sigma,d,batch_size,seed,type_of_data,order,'none',file,path)
        X,p_,g_tau = gen_func(input_gen)

        X,p_,g_tau = torch_t(X), torch_t(p_),torch_t(g_tau)
        X = torch.diff(X,1,2,prepend = torch.unsqueeze(X[:,:,0],2))
        state = torch.cat((X,p_),axis = 1)
        
        
        loss = np.zeros(1)
        loss = torch_t(loss)
        state_hist = []
        hist_state = torch.zeros((state.shape[0],state.shape[1],N)).to(device)
        for n in range(0, N, 1):
            state_for_append = torch.cat((hist_state[:,:,n+1:],torch.flip(state[:,:,:n+1],dims=[0,2])), dim=2)
            state_hist.append(torch.unsqueeze(state_for_append, dim=3))
        state_hist = torch.cat(state_hist,3)
        output_hist = []
        for n in range(N-2, -1, -1): # loop from T-T/N to T/N
            net_n = neural_net(state_hist[:,:,:,n],non_tran_var)
            F_n   = torch.sigmoid(net_n)
            loss -= torch.mean(p_[:, :, n] * F_n + g_tau * (1. - F_n)) # the loss for a single stopping time problem, we want to maximize this loss in every time step
            g_tau = torch.where(net_n > 0, p_[:, :, n], g_tau) # this is our new g_tau now, we hand it backwards in time
            

        px_mean_batch = torch.mean(g_tau)
        loss = torch.mean(loss)
        px_hist.append(px_mean_batch.item())
        
        if train_step> 10:
            if px_mean_batch.item()>max_px and px_mean_batch.item()<np.mean(px_hist[-10:-1])*1.70:
            #if np.mean(px_hist[-100:])>np.mean(px_hist[-500:]):
                #print('Best Model Saved')
                torch.save(neural_net.state_dict(), PATH+'/best_model_becker.pt')
        
        if train_step%100 == 0 or train_step <100:
            print('| Train step: {:5.0f} | Loss: {:3.3f} | Avg price: {:3.3f} | Lr: {:1.6f} |'.format(train_step,loss.item(),px_mean_batch.item(),optimizer.param_groups[0]['lr']))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    np.save(path_output+'Becker_cnn_train_'+str(N),np.asarray(px_hist))

def Becker_mod_cnn_train_model_2(s_0,K,T,N,r,delta,sigma,d,batch_size,order,type_of_data,
                     PATH,num_neurons,lr_boundaries,lr_init,training_steps,file,path,path_output):
    
    
    neural_net = Neural_Net_CNN(num_neurons,d+1,N,2,batch_size).to(device)
    optimizer = torch.optim.Adam(neural_net.parameters(), lr=0.05)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_boundaries, gamma=0.1)
    
    gen_func = choose_gen(type_of_data)
    
    K_var = torch.ones(batch_size,1).to(device)*K
    T_var = torch.ones(batch_size,1).to(device)*T
    
    non_tran_var = torch.cat((K_var,T_var),1)
    px_hist = []
    px_hist_val = []
    max_px = 0
    print('\n\n Training phase:\n\n')
    for train_step in range(0,training_steps+1):
        #Generate batch data
        seed = train_step
        torch.manual_seed(seed)
        np.random.seed(seed)

        input_gen = select_input(s_0,K,T,N,r,delta,sigma,d,batch_size,seed,type_of_data,order,'train',file,path)
        X,p_,g_tau = gen_func(input_gen)

        X,p_,g_tau = torch_t(X), torch_t(p_),torch_t(g_tau)
        X = torch.diff(X,1,2,prepend = torch.unsqueeze(X[:,:,0],2))
        state = torch.cat((X,p_),axis = 1)
        
        
        loss = np.zeros(1)
        loss = torch_t(loss)
        state_hist = []
        hist_state = torch.zeros((state.shape[0],state.shape[1],N)).to(device)
        for n in range(0, N, 1):
            state_for_append = torch.cat((hist_state[:,:,n+1:],torch.flip(state[:,:,:n+1],dims=[0,2])), dim=2)
            state_hist.append(torch.unsqueeze(state_for_append, dim=3))
        state_hist = torch.cat(state_hist,3)

        for n in range(N-2, -1, -1): # loop from T-T/N to T/N
            net_n = neural_net(state_hist[:,:,:,n],non_tran_var)
            F_n   = torch.sigmoid(net_n)
            loss -= torch.mean(p_[:, :, n] * F_n + g_tau * (1. - F_n)) # the loss for a single stopping time problem, we want to maximize this loss in every time step
            g_tau = torch.where(net_n > 0, p_[:, :, n], g_tau) # this is our new g_tau now, we hand it backwards in time
        
        px_mean_batch = torch.mean(g_tau)
        loss = torch.mean(loss)
        px_hist.append(px_mean_batch.item())
        
        #Validation: 
        with torch.no_grad():
            input_gen = select_input(s_0,K,T,N,r,delta,sigma,d,batch_size,seed,type_of_data,order,'val',file,path)
            X,p_,g_tau = gen_func(input_gen)

            X,p_,g_tau = torch_t(X), torch_t(p_),torch_t(g_tau)
            X = torch.diff(X,1,2,prepend = torch.unsqueeze(X[:,:,0],2))
            state = torch.cat((X,p_),axis = 1)
            state_hist = []
            hist_state = torch.zeros((state.shape[0],state.shape[1],N)).to(device)
            for n in range(0, N, 1):
                state_for_append = torch.cat((hist_state[:,:,n+1:],torch.flip(state[:,:,:n+1],dims=[0,2])), dim=2)
                state_hist.append(torch.unsqueeze(state_for_append, dim=3))
            state_hist = torch.cat(state_hist,3)
            for n in range(N-2, -1, -1): # loop from T-T/N to T/N
                net_n = neural_net(state_hist[:,:,:,n],non_tran_var)
                F_n   = torch.sigmoid(net_n)
                g_tau = torch.where(net_n > 0, p_[:, :, n], g_tau) # this is our new g_tau now, we hand it backwards in time
            px_mean_batch = torch.mean(g_tau)
            px_hist_val.append(px_mean_batch.item())

        
        if train_step> 5:
            if px_hist_val[-1]>max_px and px_hist_val[-1]<np.mean(px_hist_val[-5:-1])*1.70:
            #if np.mean(px_hist[-100:])>np.mean(px_hist[-500:]):
                #print('Best Model Saved')
                torch.save(neural_net.state_dict(), PATH+'/best_model_becker.pt')
        
        if train_step%100 == 0 or train_step <100:
            print('| Train step: {:5.0f} | Loss: {:3.3f} | Avg price Train: {:3.3f} | Avg price Val: {:3.3f} | Lr: {:1.6f} |'.format(train_step,loss.item(),px_hist[-1],px_hist_val[-1],optimizer.param_groups[0]['lr']))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    np.save(path_output+'Becker_cnn_train_'+str(N),np.asarray(px_hist))
    np.save(path_output+'Becker_cnn_val_'+str(N),np.asarray(px_hist_val))

