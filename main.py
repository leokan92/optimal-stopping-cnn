# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 05:02:10 2021

@author: leona

Main
"""
import torch
from torch import nn
from train_ import Becker_train_model, BeckerH_train_model,BeckerH_r_train_model,Becker_mod_cnn_train_model,Becker_mod_cnn_train_model_2
from test_ import Becker_test_model,BeckeH_test_model,BeckeH_r_test_model,Becker_mod_cnn_test_model
from utils import calculate_sigma
from LSMC import AmericanOptionsLSMC
import os

PATH = os.getcwd()
s_0,K,T,N,r,delta,sigma,d,batch_size,order,type_of_data = 100,100,3,400,0.05,0.1,0.2,1,int(8192/50),2,'real_val'
num_neurons,lr_boundaries_LSMT,lr_init_LSMT,training_steps_LSMT = 50,[200,400],0.1,500
lr_boundaries_NN,lr_init_NN,training_steps_NN,mc_runs = [100,150],0.1,150,50
path = r'C:\Users\leona\Google Drive\USP\Doutorado\PoliTO\Option Stopping\Codes\Implementation\optimal-stopping-cnn\Datasets'



#for i in range(10,500,30):
N = 30
    # Becker model using history of prices data
    
num_neurons,lr_boundaries,lr_init,training_steps,mc_runs = 50,[300,600,900],0.05,1000,2000
    
    # num_neurons,lr_boundaries,lr_init,training_steps,mc_runs = 50,[500,1000,1500],0.05,2000,500
    
    # file = r'\SP500- daily - 30Y_train.csv'
    # BeckerH_train_model(s_0,K,T,N,r,delta,sigma,d,batch_size,order,type_of_data,
    #                       PATH,num_neurons,lr_boundaries,lr_init,training_steps,path,file)
    
    # file = r'\SP500- daily - 30Y_test.csv'
    # BeckeH_test_model(s_0,K,T,N,r,delta,sigma,d,batch_size,order,type_of_data,
    #                       PATH,num_neurons,lr_boundaries_NN,lr_init,mc_runs,training_steps,path,file)
    
    
    # num_neurons,lr_boundaries,lr_init,training_steps,mc_runs = 50,[500,1000,1500],0.05,2000,500
    # file = r'\SP500- daily - 30Y_train.csv'
    # BeckerH_r_train_model(s_0,K,T,N,r,delta,sigma,d,batch_size,order,type_of_data,
    #                       PATH,num_neurons,lr_boundaries,lr_init,training_steps,path,file)
    # file = r'\SP500- daily - 30Y_test.csv'
    # BeckeH_r_test_model(s_0,K,T,N,r,delta,sigma,d,batch_size,order,type_of_data,
    #                       PATH,num_neurons,lr_boundaries_NN,lr_init,mc_runs,training_steps,path,file)
    
    # our model using history of prices returns data
    
    # num_neurons = 50
    # file = r'\SP500- daily - 30Y_train.csv'
    # Becker_mod_cnn_train_model(s_0,K,T,N,r,delta,sigma,d,batch_size,order,type_of_data,
    #                       PATH,num_neurons,lr_boundaries,lr_init,training_steps,path,file)
    # file = r'\SP500- daily - 30Y_test.csv'
    # Becker_mod_cnn_test_model(s_0,K,T,N,r,delta,sigma,d,batch_size,order,type_of_data,
    #                       PATH,num_neurons,lr_boundaries_NN,lr_init,mc_runs,training_steps,path,file)

# num_neurons = 50
# file = r'\crudeoil_train.csv'
# #file = r'\SP500- daily - 30Y_train.csv'
# Becker_mod_cnn_train_model_2(s_0,K,T,N,r,delta,sigma,d,batch_size,order,type_of_data,
#                       PATH,num_neurons,lr_boundaries,lr_init,training_steps,path,file)
# file = r'\crudeoil_test.csv'
# #file = r'\SP500- daily - 30Y_test.csv'
# Becker_mod_cnn_test_model(s_0,K,T,N,r,delta,sigma,d,batch_size,order,type_of_data,
#                       PATH,num_neurons,lr_boundaries_NN,lr_init,mc_runs,training_steps,path,file)

  

file = r'\crudeoil_train.csv'  
sigma_data = calculate_sigma(path,file)

# sigma = sigma_data
# num_neurons = 50
# file = r'\crudeoil_train.csv'
# #file = r'\SP500- daily - 30Y_train.csv'
# Becker_mod_cnn_train_model_2(s_0,K,T,N,r,delta,sigma,d,batch_size,order,type_of_data,
#                       PATH,num_neurons,lr_boundaries,lr_init,training_steps,path,file)
# file = r'\crudeoil_train.csv'
# #file = r'\SP500- daily - 30Y_test.csv'
# Becker_mod_cnn_test_model(s_0,K,T,N,r,delta,sigma,d,batch_size,order,type_of_data,
#                       PATH,num_neurons,lr_boundaries_NN,lr_init,mc_runs,training_steps,path,file)


path_type = 'brownian_motion'
path_type = 'real'
AmericanPUT = AmericanOptionsLSMC('put',s_0, K, T, N, r, delta, sigma_data, int(8192/50)*mc_runs,path_type)
print('Price: ', AmericanPUT.price())
