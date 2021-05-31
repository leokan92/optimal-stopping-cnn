# -*- coding: utf-8 -*-
"""
Created on Fri May 28 12:52:06 2021

@author: leona
"""

import pyflux

import numpy as np
import pyflux as pf 
import pandas as pd
from pandas_datareader import DataReader
from datetime import datetime
import matplotlib.pyplot as plt


path = r'C:\Users\leona\Google Drive\USP\Doutorado\PoliTO\Option Stopping\Codes\Implementation\optimal-stopping-cnn\Datasets'
file = r'\crudeoil_train.csv'  

N = 30
S0 = 100
mc_runs = 2000
runs = int(8192/50)*mc_runs
df = pd.read_csv(path+file,sep=';',thousands=',')
returns = np.diff(df['Close']) / df['Close'][:-1]

model = pf.GARCH(returns.values,p=1,q=1)
model.fit(method='BBVI', iterations=10000, optimizer='ADAM')
X = model.sample(runs)
simulations = X[:,:N].T
np.save('GARCH_SIM.npy',np.concatenate((np.expand_dims(S0*np.ones(runs),0),(S0*np.cumprod(X[:,:N].T +1,0))),0))



