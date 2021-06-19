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


################################################################################
# Simulated path experiments:
################################################################################

PATH = os.getcwd()
s_0,K,T,N,r,delta,sigma,d,batch_size,order,type_of_data = 100,100,3,400,0.05,0.1,0.2,1,int(8192/50),2,'real_val'
num_neurons,lr_boundaries_LSMT,lr_init_LSMT,training_steps_LSMT = 50,[200,400],0.1,500
lr_boundaries_NN,lr_init_NN,training_steps_NN,mc_runs = [100,150],0.1,150,50
path = r'C:\Users\leona\Google Drive\USP\Doutorado\PoliTO\Option Stopping\Codes\Implementation\optimal-stopping-cnn\Datasets'
path_output = 'Results/Energy/'

# for i in range(10,511,30):
#     N = i
#     # Becker model using history of prices data
    
#     num_neurons,lr_boundaries,lr_init,training_steps,mc_runs = 50,[100,200,300],0.05,500,2000
    
#     file = r'\SP500- daily - 30Y_train.csv'
#     BeckerH_train_model(s_0,K,T,N,r,delta,sigma,d,batch_size,order,type_of_data,
#                           PATH,num_neurons,lr_boundaries,lr_init,training_steps,path,file,path_output)
    
#     file = r'\SP500- daily - 30Y_test.csv'
#     BeckeH_test_model(s_0,K,T,N,r,delta,sigma,d,batch_size,order,type_of_data,
#                           PATH,num_neurons,lr_boundaries_NN,lr_init,mc_runs,training_steps,path,file,path_output)
    

    # our model using history of prices returns data
    
    # num_neurons = 50
    # file = r'\SP500- daily - 30Y_train.csv'
    # Becker_mod_cnn_train_model(s_0,K,T,N,r,delta,sigma,d,batch_size,order,type_of_data,
    #                       PATH,num_neurons,lr_boundaries,lr_init,training_steps,path,file,path_output)
    # file = r'\SP500- daily - 30Y_test.csv'
    # Becker_mod_cnn_test_model(s_0,K,T,N,r,delta,sigma,d,batch_size,order,type_of_data,
    #                       PATH,num_neurons,lr_boundaries_NN,lr_init,mc_runs,training_steps,path,file,path_output)


################################################################################
# Real data experiments:
################################################################################

# num_neurons = 50
# file = r'\crudeoil_train.csv'
# #file = r'\SP500- daily - 30Y_train.csv'
# Becker_mod_cnn_train_model_2(s_0,K,T,N,r,delta,sigma,d,batch_size,order,type_of_data,
#                       PATH,num_neurons,lr_boundaries,lr_init,training_steps,path,file,path_output)
# file = r'\crudeoil_test.csv'
# #file = r'\SP500- daily - 30Y_test.csv'
# Becker_mod_cnn_test_model(s_0,K,T,N,r,delta,sigma,d,batch_size,order,type_of_data,
#                       PATH,num_neurons,lr_boundaries_NN,lr_init,mc_runs,training_steps,path,file,path_output)

  

file = r'\crudeoil_train.csv'  
sigma_data = calculate_sigma(path,file)

path_output = 'Results/Energy/'
for i in range(10,511,30):
    N = i
    
    num_neurons,lr_boundaries,lr_init,training_steps,mc_runs = 50,[100,200,300],0.05,500,2000
    
    sigma = sigma_data
    num_neurons = 50
    file = r'\crudeoil_train.csv'
    #file = r'\SP500- daily - 30Y_train.csv'
    Becker_mod_cnn_train_model_2(s_0,K,T,N,r,delta,sigma,d,batch_size,order,type_of_data,
                          PATH,num_neurons,lr_boundaries,lr_init,training_steps,path,file,path_output)
    file = r'\crudeoil_test.csv'
    #file = r'\SP500- daily - 30Y_test.csv'
    Becker_mod_cnn_test_model(s_0,K,T,N,r,delta,sigma,d,batch_size,order,type_of_data,
                          PATH,num_neurons,lr_boundaries_NN,lr_init,mc_runs,training_steps,path,file,path_output)
    
    
    # #path_type = 'brownian_motion'
    # path_type = 'real'
    # AmericanPUT = AmericanOptionsLSMC('put',s_0, K, T, N, r, delta, sigma_data, int(8192/50)*mc_runs,path_type,'train')
    # print('Price: ', AmericanPUT.price())
    
################################################################################
# Exploring different assets performance:
################################################################################

# file = r'\crudeoil_train.csv'  
# sigma_data = calculate_sigma(path,file)
# asset_list = ['BCE','VZ','T','IPG','DISH','ABEV','LEN','F','BTI','GPS','BP','COP','CVX','COG','LNG','ACNB','ORI','RE','AXP','C','DD','BHP','SCCO','OLN','MOS','INTC','HPQ','AMD','LOGI','ARW']

# path_output = 'Results/Energy/'
# N = 200
# for asset in asset_list:
    
#     file    
#     num_neurons,lr_boundaries,lr_init,training_steps,mc_runs = 50,[100,200,300],0.05,500,2000
    
#     sigma = sigma_data
#     num_neurons = 50
#     file = '/'+asset+'_train.csv'
#     #file = r'\SP500- daily - 30Y_train.csv'
#     Becker_mod_cnn_train_model_2(s_0,K,T,N,r,delta,sigma,d,batch_size,order,type_of_data,
#                           PATH,num_neurons,lr_boundaries,lr_init,training_steps,path,file,path_output)
#     file = '/'+asset+'_test.csv'
#     #file = r'\SP500- daily - 30Y_test.csv'
#     Becker_mod_cnn_test_model(s_0,K,T,N,r,delta,sigma,d,batch_size,order,type_of_data,
#                           PATH,num_neurons,lr_boundaries_NN,lr_init,mc_runs,training_steps,path,file,path_output)
    
    
#     #path_type = 'brownian_motion'
#     path_type = 'real'
#     AmericanPUT = AmericanOptionsLSMC('put',s_0, K, T, N, r, delta, sigma_data, int(8192/50)*mc_runs,path_type,'train')
#     print('Price: ', AmericanPUT.price())






