# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 04:27:15 2021

@author: leona

data generation functions:
"""

import torch
import numpy as np
from stock_price_generator_heston import heston_stock_price_generation as h_gen
from sythetic_ts import NARAM_sample,AR_sample,FBM_sample,harmonic_sample,return_real_data_sample

def rolling_mean(X):
    results = []
    for n in range(1,len(X[0,0,:])+1):
        results.append(np.expand_dims(np.mean(X[:,:,:n],axis = 2),-1))
    return np.concatenate(results,axis = 2)

def generate_data_normal_dist_bermudan(u):
    s_0,K,T,N,r,delta,sigma,d,seed,batch_size = u
    torch.manual_seed(seed)
    np.random.seed(seed)
    W = np.random.normal(0, np.sqrt(T / N),size = (batch_size, d, N)) # Randon returns generate by the norma distribution
    W = np.cumsum(W, axis=2) # do a cumsum over the time dimension
    t = np.linspace(T/N, T, N) # this is the vector of out time discretization
    X = s_0 * np.exp((r - delta - sigma ** 2 / 2) * t + sigma * W) # finally we put everything together to get the price samples
    p_ = np.exp(-r*t) * np.maximum(np.max(X, axis = 1, keepdims=True) - K, 0.) # we evaluate the payoff for the whole batch at every point in time
    g_tau = p_[:,:,-1] # this is the payoff at time T, from here on we go recursivly back in time
    return X,p_,g_tau

def generate_data_heston_model(u):
    seed,N,paths,S0,K,v0,mu,rho,kappa,theta,sigma,T,r,batch_size, print_on = u
    X,Y = h_gen(seed = seed, N = N,paths = paths,S0 = S0,v0 = v0, mu = mu,
                rho = rho,kappa = kappa,theta = theta,sigma = sigma,
                batch_size = batch_size, print_on = print_on)
    t = np.linspace(T/N, T, N) # this is the vector of out time discretization
    p_ = np.exp(-r*t) * np.maximum(np.max(X, axis = 1, keepdims=True) - K, 0.)
    g_tau = p_[:,:, -1]
    return X,p_,g_tau

def generate_data_narma_model(u):
    s_0,K,T,N,r,d,batch_size,order = u
    X = NARAM_sample(s_0,K,N,d,batch_size,order)
    t = np.linspace(T/N, T, N)
    p_ = np.exp(-r*t) * np.maximum(np.max(X, axis = 1, keepdims=True) - K, 0.)
    g_tau = p_[:,:, -1]
    return X,p_,g_tau

def generate_data_ar_model(u):
    s_0,K,T,N,r,d,batch_size = u
    X = AR_sample(s_0,K,N,d,batch_size)
    t = np.linspace(T/N, T, N)
    p_ = np.exp(-r*t) * np.maximum(np.max(X, axis = 1, keepdims=True) - K, 0.)
    g_tau = p_[:,:, -1]
    return X,p_,g_tau

def generate_data_harmionic_model(u):
    s_0,K,T,N,r,d,batch_size = u
    X = harmonic_sample(s_0,K,N,d,batch_size)
    t = np.linspace(T/N, T, N)
    p_ = np.exp(-r*t) * np.maximum(np.max(X, axis = 1, keepdims=True) - K, 0.)
    g_tau = p_[:,:, -1]
    return X,p_,g_tau

def generate_data_normal_dist_asian(s_0,K,T,N,r,delta,sigma,d,seed,batch_size):
    torch.manual_seed(seed)
    np.random.seed(seed)
    W = np.random.normal(0, np.sqrt(T / N),size = (batch_size, d, N)) # Randon returns generate by the norma distribution
    W = np.cumsum(W, axis=2) # do a cumsum over the time dimension
    t = np.linspace(T/N, T, N) # this is the vector of out time discretization
    X = s_0 * np.exp((r - delta - sigma ** 2 / 2) * t + sigma * W) # finally we put everything together to get the price samples
    p_ = np.exp(-r*t) * np.maximum(np.max(rolling_mean(X), axis = 1, keepdims=True) - K, 0.) # we evaluate the payoff for the whole batch at every point in time
    g_tau = p_[:,:, -1] # this is the payoff at time T, from here on we go recursivly back in time
    return X,p_,g_tau

def generate_data_fractional_brownian_bermudan(u):
    s_0,K,T,N,r,d,batch_size,seed = u
    torch.manual_seed(seed)
    np.random.seed(seed)
    X = FBM_sample(s_0,K,N,d,batch_size)
    t = np.linspace(T/N, T, N)
    p_ = np.exp(-r*t) * np.maximum(np.max(X, axis = 1, keepdims=True) - K, 0.) # we evaluate the payoff for the whole batch at every point in time
    g_tau = p_[:,:,-1] # this is the payoff at time T, from here on we go recursivly back in time
    return X,p_,g_tau

def generate_real_data_sample(u):
    path,file,N,d,batch_size,S0,T,r,K = u
    X = return_real_data_sample(path,file,N,d,batch_size,S0)
    t = np.linspace(T/N, T, N)
    p_ = np.exp(-r*t) * np.maximum(np.max(X, axis = 1, keepdims=True) - K, 0.) # we evaluate the payoff for the whole batch at every point in time
    g_tau = p_[:,:,-1] # this is the payoff at time T, from here on we go recursivly back in time
    return X,p_,g_tau

def generate_real_data_sample_val(u):
    path,file,N,d,batch_size,S0,T,r,K,sample_type = u
    X = return_real_data_sample(path,file,N,d,batch_size,S0,sample_type)
    X_train = X[:int(len(X)*0.7)]
    t = np.linspace(T/N, T, N)
    p_train = np.exp(-r*t) * np.maximum(np.max(X_train, axis = 1, keepdims=True) - K, 0.) # we evaluate the payoff for the whole batch at every point in time
    g_tau_train = p_train[:,:,-1] # this is the payoff at time T, from here on we go recursivly back in time
    X_val = X[int(len(X)*0.7):]
    t = np.linspace(T/N, T, N)
    p_val = np.exp(-r*t) * np.maximum(np.max(X_val, axis = 1, keepdims=True) - K, 0.) # we evaluate the payoff for the whole batch at every point in time
    g_tau_val = p_train[:,:,-1] # this is the payoff at time T, from here on we go recursivly back in time
    return X,p_,g_tau

