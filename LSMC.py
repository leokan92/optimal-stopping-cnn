# -*- coding: utf-8 -*-
"""
Created on Tue May 25 01:31:03 2021

@author: leona
LSMC implementation from: https://github.com/jpcolino/IPython_notebooks/blob/master/Least%20Square%20Monte%20Carlo%20Implementation%20in%20a%20Python%20Class.ipynb

"""

import IPython
import numpy as np
import warnings
import pyflux as pf 
import pandas as pd
warnings.filterwarnings("ignore")
from sys import version 
from joblib import Parallel, delayed
import multiprocessing
# print(' Least-Squares MC for American Options: Conditions for Replication '.center(85,"-"))
# print('Python version:     ' + version)
# print('Numpy version:      ' + np.__version__)
# print('IPython version:    ' + IPython.__version__)
# print('-'*85)

import numpy as np

class AmericanOptionsLSMC(object):
    """ Class for American options pricing using Longstaff-Schwartz (2001):
    "Valuing American Options by Simulation: A Simple Least-Squares Approach."
    Review of Financial Studies, Vol. 14, 113-147.
    S0 : float : initial stock/index level
    strike : float : strike price
    T : float : time to maturity (in year fractions)
    M : int : grid or granularity for time (in number of total points)
    r : float : constant risk-free short rate
    div :    float : dividend yield
    sigma :  float : volatility factor in diffusion term 
    
    Unitest(doctest): 
    >>> AmericanPUT = AmericanOptionsLSMC('put', 36., 40., 1., 50, 0.06, 0.06, 0.2, 10000  )
    >>> AmericanPUT.price
    4.4731177017712209
    """

    def __init__(self, option_type, S0, strike, T, M, r, div, sigma, simulations,path_type):
        try:
            self.option_type = option_type
            assert isinstance(option_type, str)
            self.S0 = float(S0)
            self.strike = float(strike)
            assert T > 0
            self.T = float(T)
            assert M > 0
            self.M = int(M)
            assert r >= 0
            self.r = float(r)
            assert div >= 0
            self.div = float(div)
            assert sigma > 0
            self.sigma = float(sigma)
            assert simulations > 0
            self.simulations = int(simulations)
            self.path_type = path_type
        except ValueError:
            print('Error passing Options parameters')


        if option_type != 'call' and option_type != 'put':
            raise ValueError("Error: option type not valid. Enter 'call' or 'put'")
        if S0 < 0 or strike < 0 or T <= 0 or r < 0 or div < 0 or sigma < 0:
            raise ValueError('Error: Negative inputs not allowed')

        self.time_unit = self.T / float(self.M)
        self.discount = np.exp(-self.r * self.time_unit)

    # def MCprice_matrix(self, seed = 123):
    #     """ Returns MC price matrix rows: time columns: price-path simulation """
    #     np.random.seed(seed)
    #     MCprice_matrix = np.zeros((self.M + 1, self.simulations), dtype=np.float64)
    #     MCprice_matrix[0,:] = self.S0
    #     for t in range(1, self.M + 1):
    #         brownian = np.random.standard_normal(int(self.simulations / 2))
    #         brownian = np.concatenate((brownian, -brownian))
    #         MCprice_matrix[t, :] = (MCprice_matrix[t - 1, :]
    #                               * np.exp((self.r - self.sigma ** 2 / 2.) * self.time_unit
    #                               + self.sigma * brownian * np.sqrt(self.time_unit)))
    #     return MCprice_matrix
    
    
    def MCprice_matrix(self, seed = 123):
        """ Returns MC price matrix rows: time columns: price-path simulation """
        np.random.seed(seed)
        MCprice_matrix = np.zeros((self.M + 1, self.simulations), dtype=np.float64)
        MCprice_matrix[0,:] = self.S0
        if self.path_type == 'garch':
            print('GARCH simulation')
            path = r'C:\Users\leona\Google Drive\USP\Doutorado\PoliTO\Option Stopping\Codes\Implementation\optimal-stopping-cnn\Datasets'
            file = r'\GARCH_SIM.npy'
            MCprice_matrix = np.load(path+file)
            # file = r'\crudeoil_train.csv'  
            # df = pd.read_csv(path+file,sep=';',thousands=',')
            # returns = np.diff(df['Close']) / df['Close'][:-1]
            # model = pf.GARCH(returns.values,p=1,q=1)
            # model.fit(method='BBVI', iterations=10000, optimizer='ADAM')
            # X = model.sample(self.simulations)
            # MCprice_matrix = np.concatenate((np.expand_dims(self.S0*np.ones(self.simulations),0),(self.S0*np.cumprod(X[:,:self.M].T +1,0))),0)
        if self.path_type == 'real':
            path = r'C:\Users\leona\Google Drive\USP\Doutorado\PoliTO\Option Stopping\Codes\Implementation\optimal-stopping-cnn\Datasets'
            file = r'\crudeoil_test.csv'
            num_cores = multiprocessing.cpu_count()
            df = pd.read_csv(path+file,sep=';',thousands=',')
            #df = pd.read_csv(path+file,sep=',')
            returns = np.diff(df['Close']) / df['Close'][:-1]
            def rand_sample(i):
                asset_list = []
                d=1
                for j in range(0,d):
                    rand = np.random.randint(0,len(returns)-self.M)
                    asset_list.append(np.concatenate((np.array([self.S0]),self.S0*np.cumprod(1+returns[rand:rand+self.M].to_numpy()))))
                return asset_list
            results = Parallel(n_jobs=num_cores)(delayed(rand_sample)(i) for i in range(0,self.simulations))
            MCprice_matrix = np.squeeze(np.asarray(results).T,1)
        if self.path_type=='brownian_motion':
            print('Brownian Motion Simulation')
            for t in range(1, self.M + 1):
                brownian = np.random.standard_normal(int(self.simulations / 2))
                brownian = np.concatenate((brownian, -brownian))
                MCprice_matrix[t, :] = (MCprice_matrix[t - 1, :]
                                      * np.exp((self.r - self.sigma ** 2 / 2.) * self.time_unit
                                      + self.sigma * brownian * np.sqrt(self.time_unit)))
        return MCprice_matrix

    def MCpayoff(self):
        """Returns the inner-value of American Option"""
        if self.option_type == 'call':
            payoff = np.maximum(self.MCprice_matrix() - self.strike,
                           np.zeros((self.M + 1, self.simulations),dtype=np.float64))
        else:
            payoff = np.maximum(self.strike - self.MCprice_matrix(),
                            np.zeros((self.M + 1, self.simulations),
                            dtype=np.float64))
        return payoff

    def value_vector(self):
        value_matrix = np.zeros_like(self.MCpayoff())
        value_matrix[-1, :] = self.MCpayoff()[-1, :]
        for t in range(self.M - 1, 0 , -1):
            regression = np.polyfit(self.MCprice_matrix()[t, :], value_matrix[t + 1, :] * self.discount, 5)
            continuation_value = np.polyval(regression, self.MCprice_matrix()[t, :])
            value_matrix[t, :] = np.where(self.MCpayoff()[t, :] > continuation_value,
                                          self.MCpayoff()[t, :],
                                          value_matrix[t + 1, :] * self.discount)
        px_vec = value_matrix[1,:] * self.discount
        np.save('Results/'+'LSMC_'+str(self.M),np.asarray(px_vec))
        
        return value_matrix[1,:] * self.discount


    def price(self): return np.sum(self.value_vector()) / float(self.simulations)
    
    def delta(self):
        diff = self.S0 * 0.01
        myCall_1 = AmericanOptionsLSMC(self.option_type, self.S0 + diff, 
                                        self.strike, self.T, self.M, 
                                        self.r, self.div, self.sigma, self.simulations)
        myCall_2 = AmericanOptionsLSMC(self.option_type, self.S0 - diff, 
                                        self.strike, self.T, self.M, 
                                        self.r, self.div, self.sigma, self.simulations)
        return (myCall_1.price() - myCall_2.price()) / float(2. * diff)
    

    def gamma(self):
        diff = self.S0 * 0.01
        myCall_1 = AmericanOptionsLSMC(self.option_type, self.S0 + diff, 
                                        self.strike, self.T, self.M, 
                                        self.r, self.div, self.sigma, self.simulations)
        myCall_2 = AmericanOptionsLSMC(self.option_type, self.S0 - diff, 
                                        self.strike, self.T, self.M, 
                                        self.r, self.div, self.sigma, self.simulations)
        return (myCall_1.delta() - myCall_2.delta()) / float(2. * diff)
    

    def vega(self):
        diff = self.sigma * 0.01
        myCall_1 = AmericanOptionsLSMC(self.option_type, self.S0, 
                                        self.strike, self.T, self.M, 
                                        self.r, self.div, self.sigma + diff, 
                                        self.simulations)
        myCall_2 = AmericanOptionsLSMC(self.option_type, self.S0,
                                        self.strike, self.T, self.M, 
                                        self.r, self.div, self.sigma - diff, 
                                        self.simulations)
        return (myCall_1.price() - myCall_2.price()) / float(2. * diff)    
    

    def rho(self):        
        diff = self.r * 0.01
        if (self.r - diff) < 0:        
            myCall_1 = AmericanOptionsLSMC(self.option_type, self.S0, 
                                        self.strike, self.T, self.M, 
                                        self.r + diff, self.div, self.sigma, 
                                        self.simulations)
            myCall_2 = AmericanOptionsLSMC(self.option_type, self.S0, 
                                        self.strike, self.T, self.M, 
                                        self.r, self.div, self.sigma, 
                                        self.simulations)
            return (myCall_1.price() - myCall_2.price()) / float(diff)
        else:
            myCall_1 = AmericanOptionsLSMC(self.option_type, self.S0, 
                                        self.strike, self.T, self.M, 
                                        self.r + diff, self.div, self.sigma, 
                                        self.simulations)
            myCall_2 = AmericanOptionsLSMC(self.option_type, self.S0, 
                                        self.strike, self.T, self.M, 
                                        self.r - diff, self.div, self.sigma, 
                                        self.simulations)
            return (myCall_1.price() - myCall_2.price()) / float(2. * diff)
    

    def theta(self): 
        diff = 1 / 252.
        myCall_1 = AmericanOptionsLSMC(self.option_type, self.S0, 
                                        self.strike, self.T + diff, self.M, 
                                        self.r, self.div, self.sigma, 
                                        self.simulations)
        myCall_2 = AmericanOptionsLSMC(self.option_type, self.S0, 
                                        self.strike, self.T - diff, self.M, 
                                        self.r, self.div, self.sigma, 
                                        self.simulations)
        return (myCall_2.price() - myCall_1.price()) / float(2. * diff)
    